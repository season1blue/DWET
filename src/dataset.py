"""
    -----------------------------------
    dataset of nel
"""
import time

import torch
from torch.utils.data import Dataset
import h5py
import json
import re
import random
from random import choice
import numpy as np
import os

from os.path import join, exists

INAME_PATTERN = re.compile("(\d+)\.")
num_sample = 0

class NELDataset(Dataset):
    def __init__(self, args, features, contain_search_res=False):
        self.features = features
        self.max_sample_num = args.neg_sample_num
        neg_config = json.load(open(args.path_neg_config))
        self.neg_iid = neg_config["neg_iid"]
        self.tfidf_neg = neg_config["tfidf_neg"]
        self.negid2qid = neg_config["keys_ordered"]
        self.qid2negid = {qid: i for i, qid in enumerate(neg_config["keys_ordered"])}

        # Sample features of negative sampling
        self.neg_list = json.load(open(join(args.dir_neg_feat, "entity_list.json")))  # len = 25846
        self.neg_mapping = {sample: i for i, sample in enumerate(self.neg_list)}

        entity_feat = h5py.File(join(args.dir_neg_feat, "entity_clip_{}.h5".format(args.gt_type)), 'r')
        self.entity_features = entity_feat.get("features")

        # search candidates
        self.contain_search_res = contain_search_res
        if self.contain_search_res:
            self.search_res = json.load(
                open(args.path_candidates, "r", encoding='utf8'))   # mention: [qid0, qid1, ..., qidn]
        

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        # feature keys: answer_id, img_id, mentions, key_id, text_feature, mention_feature,total_feature, segement_feature, profile_feature
        
        sample = dict()
        sample["mention_feature"] = feature.mention_feature
        sample["text_feature"] = feature.text_feature
        sample['total_feature'] = feature.total_feature
        sample['profile_feature'] = feature.profile_feature
        sample['segement_feature'] = feature.segement_feature
        sample['identity_feature'] = feature.identity_feature if "identity_feature" in sample.keys() else feature.profile_feature

        ans_id = "Q5729149" if feature.answer_id == "c" else feature.answer_id

        pos_sample_id = self.neg_mapping[ans_id]
        neg_ids = neg_sample_online(self.qid2negid[ans_id], self.neg_iid, self.tfidf_neg, self.negid2qid,
                                    self.max_sample_num)
        neg_ids_map = [self.neg_mapping[nid] for nid in neg_ids]

        sample["pos_sample"] = torch.tensor(np.array([self.entity_features[pos_sample_id]]))
        sample["neg_sample"] = torch.tensor(np.array([self.entity_features[nim] for nim in neg_ids_map]))

        # return search results
        if self.contain_search_res:
            qids_searched = self.search_res[feature.mentions]
            qids_searched_map = [self.neg_mapping[qid] for qid in qids_searched]
            sample["search_res"] = torch.tensor(np.array([self.entity_features[qsm] for qsm in qids_searched_map]))
            # print(sample["search_res"].size())  #Bathc_size*hidden_size 32*768
        return sample



class DiverseDataset(Dataset):
    def __init__(self, args, all_text_feature, all_mention_feature, all_total_feature, all_profile_feature, all_answer_id, contain_search_res):
        self.args = args
        self.all_answer_id = all_answer_id
        self.all_text_feature = all_text_feature
        self.all_mention_feature = all_mention_feature
        self.all_total_feature = all_total_feature
        self.all_profile_feature = all_profile_feature
        self.max_sample_num = args.neg_sample_num

        self.entity_list = json.load(open(join(args.dir_neg_feat, "entity_list.json")))  # len = 25846
        self.entity_id_mapping = {sample: i for i, sample in enumerate(self.entity_list)}

        entity_feat = h5py.File(join(args.dir_neg_feat, "entity_clip_plus.h5"), 'r')
        self.entity_features = entity_feat.get("features")

        self.contain_search_res = contain_search_res
        self.neg_sample_list = json.load(open(join(args.dir_neg_feat, "search_top100.json"), "r", encoding='utf8'))

    def __len__(self):
        return len(self.all_answer_id)

    def __getitem__(self, idx):
        sample = dict()
        ans_qid = self.all_answer_id[idx]  # Prince Harry
        pos_sample_id = self.entity_id_mapping[ans_qid]  # 46729
        neg_sample_qids = self.neg_sample_list[ans_qid][:min(len(self.neg_sample_list[ans_qid]),
                                                             self.max_sample_num)]  # ['Educational background of George W. Bush']
        neg_sample_ids = [self.entity_id_mapping[qids] for qids in
                          neg_sample_qids]  # [46729, 25088, 32776, 116770, 96808.. ] 100

        sample["answer_id"] = self.entity_id_mapping[self.all_answer_id[idx]]
        sample['text_feature'] = self.all_text_feature[idx]
        sample['mention_feature'] = self.all_mention_feature[idx]
        sample['total_feature'] = self.all_total_feature[idx]
        sample['profile_feature'] = self.all_profile_feature[idx]

        sample["pos"] = torch.tensor(np.array([self.entity_features[pos_sample_id]]))
        sample["neg"] = torch.tensor(np.array([self.entity_features[nim] for nim in neg_sample_ids]))

        if self.contain_search_res:
            qids_searched = self.neg_sample_list[ans_qid]
            qids_searched_map = [pos_sample_id] + [self.entity_id_mapping[qid] for qid in qids_searched]
            sample["search_res"] = torch.tensor(np.array([self.entity_features[qsm] for qsm in qids_searched_map]))
            sample["search_res"] = sample["search_res"][:80]  #TODO

        return sample


class PersonDataset(Dataset):
    def __init__(self, args, all_img_id, all_answer_id, all_image_feature, contain_search_res):
        self.args = args
        self.all_img_id = all_img_id
        self.all_answer_id = all_answer_id
        self.all_image_features = all_image_feature
        self.max_sample_num = args.neg_sample_num

        self.entity_list = json.load(open(join(args.dir_neg_feat, "entity_list.json")))  # len = 25846
        self.entity_id_mapping = {sample: i for i, sample in enumerate(self.entity_list)}
        self.neg_config = json.load(open(args.path_neg_config))
        self.qid2negid = {qid: i for i, qid in enumerate(self.neg_config["keys_ordered"])}

        entity_list = json.load(open(self.args.path_ans_list))
        self.entity_mapping = {sample: i for i, sample in enumerate(entity_list)}

        img_list = json.load(open(join(args.dir_neg_feat, "input_img_list.json")))
        self.img_mapping = {sample: i for i, sample in enumerate(img_list)}

        caption_path = os.path.join(args.dir_neg_feat, "caption.h5")
        self.caption = h5py.File(caption_path, 'r').get("features")

        if args.gt_type != "both":
            gt_name = "{}_entity.h5".format(args.gt_type)
            entity_feat = h5py.File(join(args.dir_neg_feat, gt_name), 'r')
            self.entity_features = entity_feat.get("features")
        else:
            entity_image_feat = h5py.File(join(args.dir_neg_feat, "image_entity.h5"), 'r')
            entity_text_feat = h5py.File(join(args.dir_neg_feat, "text_entity.h5"), 'r')
            self.visual_entity_features = entity_image_feat.get("features")
            self.textual_entity_features = entity_text_feat.get("features")

        self.contain_search_res = contain_search_res
        self.neg_sample_list = json.load(open(args.path_candidates, "r", encoding='utf8'))

    def __len__(self):
        return len(self.all_answer_id)

    def __getitem__(self, idx):
        sample = dict()

        img_id = self.all_img_id[idx]
        ans_qid = self.all_answer_id[idx]  # Q3290309
        pos_sample_id = self.entity_id_mapping[ans_qid]  # 46729
        neg_sample_qids = self.neg_sample_list[ans_qid][:min(len(self.neg_sample_list[ans_qid]),
                                                             self.max_sample_num)]  # ['Q12586851', 'Q2929059', 'Q4720236', 'Q19958130'
        neg_sample_ids = [self.entity_id_mapping[qids] for qids in
                          neg_sample_qids]  # [46729, 25088, 32776, 116770, 96808.. ] 100

        sample["answer_id"] = self.entity_mapping[self.all_answer_id[idx]]
        sample['image_feature'] = self.all_image_features[idx]
        sample["detection"] = torch.tensor(np.array([self.caption[self.img_mapping[img_id]]]))

        if self.args.gt_type != "both":
            sample["pos"] = torch.tensor(np.array([self.entity_features[pos_sample_id]]))
            sample["neg"] = torch.tensor(np.array([self.entity_features[nim] for nim in neg_sample_ids]))
        else:
            pos_textual_feature = torch.tensor(np.array([self.textual_entity_features[pos_sample_id]]))
            pos_visual_feature = torch.tensor(np.array([self.visual_entity_features[pos_sample_id]]))
            neg_textual_feature = torch.tensor(np.array([self.textual_entity_features[nim] for nim in neg_sample_ids]))
            neg_visual_feature = torch.tensor(np.array([self.visual_entity_features[nim] for nim in neg_sample_ids]))

            sample["pos"] = pos_textual_feature + pos_visual_feature
            sample["neg"] = neg_textual_feature + neg_visual_feature

        if self.contain_search_res:
            qids_searched = self.neg_sample_list[ans_qid]
            qids_searched_map = [pos_sample_id] + [self.entity_id_mapping[qid] for qid in qids_searched]
            sample["search_res"] = torch.tensor(np.array([self.entity_features[qsm] for qsm in qids_searched_map]))
        return sample





def train_collate_fn(batch):
    search_res = torch.stack([b["search_res"] for b in batch]) if "search_res" in batch[0].keys() else None
    
    mention_feature = torch.stack([b["mention_feature"] for b in batch])
    total_feature = torch.stack([b["total_feature"] for b in batch])
    text_feature = torch.stack([b["text_feature"] for b in batch])

    segement_feature_list = [b["segement_feature"] for b in batch]
    profile_feature_list = [b["profile_feature"] for b in batch]
    identity_feature_list = [b["identity_feature"] for b in batch]
    pos_sample_list = [b["pos_sample"] for b in batch]
    neg_sample_list = [b["neg_sample"] for b in batch]
    

    # TODO: in-batch negatives
    for index, b in enumerate(batch):
        for times in range(num_sample):
            rand = choice([i for i in range(0, len(pos_sample_list) - 1) if i != index])
            neg = pos_sample_list[rand]
            neg_sample_list[index] = torch.cat([neg_sample_list[index], neg], dim=0)

    pos_sample = torch.stack(pos_sample_list)
    neg_sample = torch.stack(neg_sample_list)

    max_size = max([imf.size(0) for imf in segement_feature_list])  # img_feature.size == (n, 512)

    for imf_index in range(len(segement_feature_list)):
        # if segement_feature_list[imf_index].size()!=identity_feature_list[imf_index].size():
        #     print(segement_feature_list[imf_index].size(), identity_feature_list[imf_index].size())
        while segement_feature_list[imf_index].size(0) < max_size:
            segement_feature_list[imf_index] = torch.nn.functional.pad(segement_feature_list[imf_index], pad=(0, 0, 0, 1), mode='constant', value=0)
            profile_feature_list[imf_index] = torch.nn.functional.pad(profile_feature_list[imf_index], pad=(0, 0, 0, 1), mode='constant', value=0)
            identity_feature_list[imf_index] = torch.nn.functional.pad(identity_feature_list[imf_index], pad=(0, 0, 0, 1), mode='constant', value=0)
    

    segement_feature = torch.stack(segement_feature_list)
    profile_feature = torch.stack(profile_feature_list)
    identity_feature = torch.stack(identity_feature_list)

    return {
        "identity_feature":identity_feature,
        "mention_feature": mention_feature,
        "text_feature": text_feature,
        "total_feature": total_feature,
        "segement_feature": segement_feature,
        "profile_feature": profile_feature,
        "pos": pos_sample,
        "neg": neg_sample,
        "search_res": search_res,
    }



def neg_sample_online(neg_id, neg_iid, tfidf_neg, negid2qid, max_sample_num=1, threshold=0.95):
    """
        Online negative sampling algorithm
        ------------------------------------------
        Args:
        Returns:
    """
    N = len(tfidf_neg)
    cands = set()

    while len(cands) < max_sample_num:
        rand = random.random()
        # print("neg id", neg_id, tfidf_neg[neg_id])
        if not tfidf_neg[neg_id] or rand > threshold:
            cand = random.randint(0, N - 1)
        else:
            rand_word = random.choice(tfidf_neg[neg_id])
            cand = random.choice(neg_iid[rand_word])

        if cand != neg_id:
            cands.add(cand)

    return [negid2qid[c] for c in cands]


def neg_sample(entity_list, pos_id, max_sample_num):
    candidate = set()
    while len(candidate) < max_sample_num:
        rand = random.randint(0, len(entity_list) - 1)  # randint [0,x] 闭区间
        if rand != pos_id:
            candidate.add(rand)
    return list(candidate)


def search_res(entity_list, pos_id, max_sample_num=1000):
    candidate = random.sample(range(0, len(entity_list) - 1), max_sample_num)
    candidate = [pos_id] + candidate
    return candidate

