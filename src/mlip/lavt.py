import torch
import torch.nn as nn
import torch.nn.functional as F


class LAVT(nn.Module):
    def __init__(self,
                 dim,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 num_heads_fusion=1,
                 fusion_drop=0.0
                 ):
        super().__init__()

        # fuse before downsampling
        self.fusion = PWAM(dim,  # both the visual input and for combining, num of channels
                           dim,  # v_in
                           768,  # l_in
                           dim,  # key
                           dim,  # value
                           num_heads=num_heads_fusion,
                           dropout=fusion_drop)

        self.res_gate = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),
            nn.Tanh()
        )
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
        # initialize the gate to 0
        nn.init.zeros_(self.res_gate[0].weight)
        nn.init.zeros_(self.res_gate[2].weight)

    def forward(self, img_feat, img_height, img_width, embedding, emb_mask):
        """ Forward function.
        Args:
            img_height: 高
            img_width: 宽
            emb_mask: embedding mask
            img_feat: Input feature, tensor size (B, H*W, C).
            embedding: embedding    l_mask:attention
        """

        # PWAM fusion
        x_residual = self.fusion(img_feat, embedding, emb_mask)
        # apply a gate on the residual
        img_feat = img_feat + (self.res_gate(x_residual) * x_residual)

        if self.downsample is not None:
            x_down = self.downsample(img_feat, img_height, img_width)
            Wh, Ww = (img_height + 1) // 2, (img_width + 1) // 2
            return x_residual, img_height, img_width, x_down, Wh, Ww
        else:
            return x_residual, img_height, img_width, img_feat, img_height, img_width


class PWAM(nn.Module):
    def __init__(self, dim, v_in_channels, l_in_channels, key_channels, value_channels, num_heads=0, dropout=0.0):
        super(PWAM, self).__init__()
        # input x shape: (B, H*W, dim)
        self.vis_project = nn.Sequential(nn.Conv1d(dim, dim, 1, 1),  # the init function sets bias to 0 if bias is True
                                         nn.GELU(),
                                         nn.Dropout(dropout)
                                        )

        self.image_lang_att = SpatialImageLanguageAttention(v_in_channels,  # v_in
                                                            l_in_channels,  # l_in
                                                            key_channels,  # key
                                                            value_channels,  # value
                                                            out_channels=value_channels,  # out
                                                            num_heads=num_heads)

        self.project_mm = nn.Sequential(nn.Conv1d(value_channels, value_channels, 1, 1),
                                        nn.GELU(),
                                        nn.Dropout(dropout)
                                        )

    def forward(self, x, l, l_mask):
        # input x shape: (B, H*W, dim)
        vis = self.vis_project(x.permute(0, 2, 1))  # (B, dim, H*W)

        lang = self.image_lang_att(x, l, l_mask)  # (B, H*W, dim)

        lang = lang.permute(0, 2, 1)  # (B, dim, H*W)

        mm = torch.mul(vis, lang)
        mm = self.project_mm(mm)  # (B, dim, H*W)

        mm = mm.permute(0, 2, 1)  # (B, H*W, dim)

        return mm


class SpatialImageLanguageAttention(nn.Module):
    def __init__(self, v_in_channels, l_in_channels, key_channels, value_channels, out_channels=None, num_heads=1):
        super(SpatialImageLanguageAttention, self).__init__()
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        self.v_in_channels = v_in_channels
        self.l_in_channels = l_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.num_heads = num_heads
        if out_channels is None:
            self.out_channels = self.value_channels

        # Keys: language features: (B, l_in_channels, #words)
        # avoid any form of spatial normalization because a sentence contains many padding 0s
        self.f_key = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.key_channels, kernel_size=1, stride=1),
        )

        # Queries: visual features: (B, H*W, v_in_channels)
        self.f_query = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.key_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.key_channels),
        )

        # Values: language features: (B, l_in_channels, #words)
        self.f_value = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.value_channels, kernel_size=1, stride=1),
        )

        # Out projection
        self.W = nn.Sequential(
            nn.Conv1d(self.value_channels, self.out_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.out_channels),
        )

    def forward(self, x, l, l_mask):
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        B, HW = x.size(0), x.size(1)
        x = x.permute(0, 2, 1)  # (B, key_channels, H*W)
        l_mask = l_mask.permute(0, 2, 1)  # (B, N_l, 1) -> (B, 1, N_l)

        query = self.f_query(x)  # (B, key_channels, H*W) if Conv1D
        query = query.permute(0, 2, 1)  # (B, H*W, key_channels)
        key = self.f_key(l)  # (B, key_channels, N_l)
        value = self.f_value(l)  # (B, self.value_channels, N_l)
        key = key * l_mask  # (B, key_channels, N_l)
        value = value * l_mask  # (B, self.value_channels, N_l)
        n_l = value.size(-1)
        query = query.reshape(B, HW, self.num_heads, self.key_channels//self.num_heads).permute(0, 2, 1, 3)
        # (b, num_heads, H*W, self.key_channels//self.num_heads)
        key = key.reshape(B, self.num_heads, self.key_channels//self.num_heads, n_l)
        # (b, num_heads, self.key_channels//self.num_heads, n_l)
        value = value.reshape(B, self.num_heads, self.value_channels//self.num_heads, n_l)
        # # (b, num_heads, self.value_channels//self.num_heads, n_l)
        l_mask = l_mask.unsqueeze(1)  # (b, 1, 1, n_l)

        sim_map = torch.matmul(query, key)  # (B, self.num_heads, H*W, N_l)
        sim_map = (self.key_channels ** -.5) * sim_map  # scaled dot product

        sim_map = sim_map + (1e4*l_mask - 1e4)  # assign a very small number to padding positions
        sim_map = F.softmax(sim_map, dim=-1)  # (B, num_heads, h*w, N_l)
        out = torch.matmul(sim_map, value.permute(0, 1, 3, 2))  # (B, num_heads, H*W, self.value_channels//num_heads)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, HW, self.value_channels)  # (B, H*W, value_channels)
        out = out.permute(0, 2, 1)  # (B, value_channels, HW)
        out = self.W(out)  # (B, value_channels, HW)
        out = out.permute(0, 2, 1)  # (B, HW, value_channels)

        return out