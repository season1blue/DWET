import logging
import os
import json
import torch
from PIL import Image
import clip
from tqdm import tqdm

from clip import load as clip_load
from clip import tokenize as clip_tokenize
from pixellib.torchbackend.instance import instanceSegmentation

from src.input_format import InputExample, InputFeatures


class Wikipedia():
    def __init__(self, args):
        super(Wikipedia, self).__init__()

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
        model_path = os.path.join(args.pretrain_model_path, "ViT-B-32.pt")
        model, preprocess = clip_load(model_path, device=self.device, jit=False)
        self.model = model
        self.preprocess = preprocess

        self.img_path = args.img_path
        Image.MAX_IMAGE_PIXELS = 2300000000  # 更改阈值像素上限

        self.cache_path = args.cache_path

        # segment
        self.ins = instanceSegmentation()
        self.ins.load_model(args.seg_model_path)
        self.target_classes = self.ins.select_target_classes(person=True)
        self.detection_path = os.path.join(args.data_dir, "wiki_detection")
        self.segement_path = os.path.join(args.data_dir, "wiki_segement")
        self.total2part_map = json.load(open(os.path.join(args.data_dir, "total2part_map.json"), 'r'))

    def read_examples_from_file(self, data_dir, mode):
        file_path = os.path.join(data_dir, "{}.json".format(mode))
        examples = []

        js = json.load(open(file_path, encoding="utf-8"))

        for k, v in js.items():
            examples.append(
                InputExample(
                    guk=k,  # f"{mode}-{k}",
                    sent=v["sentence"],
                    idx=k,  # v["id"]
                    answer=v["answer"],  # v["answer"]
                    mentions=v["mentions"],
                    img_path=v['imgPath']
                )
            )

        return examples

    # segement image to person image
    def split_image(self, img_path):
        # make sure delete the past segement result
        if len(os.listdir(self.cache_path)) != 0:
            for file in os.listdir(self.cache_path):
                os.remove(os.path.join(self.cache_path, file))

        self.ins.segmentImage(img_path, show_bboxes=True, extract_segmented_objects=True,
                              segment_target_classes=self.target_classes, save_extracted_objects=True,
                              output_path=self.cache_path)

        image_list = []
        for file in os.listdir(self.cache_path):
            file_path = os.path.join(self.cache_path, file)
            image = Image.open(file_path)
            image = self.preprocess(image)
            image = image.unsqueeze(0).to(self.device)
            image_list.append(image)
        return image_list

    def clip_feature(self, examples):
        features = []
        for (ex_index, example) in tqdm(enumerate(examples), total=len(examples), ncols=80):
            self.model.to(self.device)
            img_id = example.img_path.split("/")[-1].split(".")[0]
            with torch.no_grad():
                input_sent = example.mentions + " [SEP] " + example.sent
                sent_ids = clip_tokenize(input_sent, truncate=True).to(self.device)  # 截断过长的
                mention = clip_tokenize(example.mentions, truncate=True).to(self.device)

                text_feature = self.model.encode_text(sent_ids)  # text_features 1,512
                mention_feature = self.model.encode_text(mention)

                # extract image feature (split or single)
                img_path = os.path.join(self.img_path, example.img_path)

                total_image = self.preprocess(Image.open(img_path)).unsqueeze(0).to(self.device)
                total_feature = self.model.encode_image(total_image)

                if img_id not in self.total2part_map:
                    segement_feature = torch.zeros_like(text_feature)
                    profile_feature = torch.zeros_like(text_feature)
                    identity_feature = torch.zeros_like(text_feature)
                else:
                    segement_list, profile_list, identity_list = [], [], []
                    for part in self.total2part_map[img_id]:
                        # segement image feature extraction
                        segement_path = os.path.join(self.segement_path, part + ".jpg")
                        segement = self.preprocess(Image.open(segement_path)).unsqueeze(0).to(self.device)
                        segement_feature = self.model.encode_image(segement)
                        segement_list.append(segement_feature)

                        # detection profile feature extraction
                        detection_path = os.path.join(self.detection_path, part + ".json")
                        facial_context = json.load(open(detection_path, 'r'))[0]
                        
                        if "dominant_gender" not in facial_context.keys():
                            profile_list.append(torch.zeros_like(text_feature))
                        else:
                            gender, race, age, emotion = facial_context['dominant_gender'], facial_context['dominant_race'], facial_context['age'], facial_context['dominant_emotion']
                            profile = "gender: {}, race: {}, age: {}, emotion: {}".format(gender, race, age, emotion)
                            profile = clip_tokenize(profile, truncate=True).to(self.device)
                            profile_feature = self.model.encode_text(profile)
                            profile_list.append(profile_feature)
                        
                        
                        identity = json.load(open(detection_path, 'r'))
                        i_result = []
                        for i in identity:
                            if 'score' in i.keys() and i['score'] > 0.5:
                                i_result.append(i['label'])
                        identity = ", ".join(i_result)
                        identity = clip_tokenize(identity, truncate=True).to(self.device)
                        identity_feature = self.model.encode_text(identity) if len(i_result)!=0 else torch.zeros_like(text_feature)   
                        identity_list.append(identity_feature)

                    segement_feature = torch.cat(segement_list, dim=0)
                    profile_feature = torch.cat(profile_list, dim=0)
                    identity_feature = torch.cat(identity_list, dim=0)

            answer_id = example.answer if example.answer else -1

            features.append(
                InputFeatures(
                    answer_id=example.answer if example.answer else -1,
                    mentions=example.mentions,
                    text_feature=text_feature,
                    mention_feature=mention_feature,
                    total_feature=total_feature,
                    segement_feature=segement_feature,
                    profile_feature=profile_feature,
                    identity_feature=identity_feature
                )
            )
        return features
