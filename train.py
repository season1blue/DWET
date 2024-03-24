import os
import json
import logging
import numpy as np
import torch
import random
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from transformers.models.auto.modeling_auto import MODEL_MAPPING
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,
)

from src.dataset import NELDataset, train_collate_fn
from src.nel import NELModel
from src.metric_topk import cal_top_k
from time import time
from src.args import parse_arg
from src.dataset import PersonDataset, NELDataset


from src.prepare_wikipedia import Wikipedia 
from src.prepare_richpedia import Richpedia
from src.prepare_wikiperson import Wikiperson
# from src.prepare_wikidiverse import Wikidiverse



# 1、创建一个logger
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)
# 2、创建一个handler，用于写入日志文件
fh = logging.FileHandler('../log.log')
fh.setLevel(logging.DEBUG)
# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# 3、定义handler的输出格式（formatter）
formatter = logging.Formatter('%(asctime)s:  %(message)s', datefmt="%m/%d %H:%M:%S")
# 4、给handler添加formatter
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# 5、给logger添加handler
logger.addHandler(fh)
logger.addHandler(ch)

# 忽略not init权重的warning提示
from transformers import logging

logging.set_verbosity_error()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)





def load_and_cache_examples(args, mode, dataset="wiki", logger=None):
    # 0 caption 没有输入prompt
    number = 1 # question = "Question:  Who are the characters in the picture? Answer: "
    cached_features_file = os.path.join(args.data_dir, "cached_{}_{}".format(mode, args.dataset))

    data_processor_mapping = {"wiki": Wikipedia, "rich": Richpedia, "person" : Wikiperson}
    if mode != 'test' and os.path.exists(cached_features_file) and not args.overwrite_cache:
        features = torch.load(cached_features_file, map_location='cuda:0')
    else:
        data_processor = data_processor_mapping[dataset](args)
        logger.info("Creating features %s at %s" % (cached_features_file, args.data_dir))
        data = data_processor.read_examples_from_file(args.data_dir, mode)
        features = data_processor.clip_feature(data) 

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    contain_search_res = False if mode == "train" else True

    if args.dataset == "person":
        all_img_id = [f.img_id for f in features]
        all_answer_id = [f.answer for f in features]
        all_image_feature = [f.image_feature for f in features]
        dataset = PersonDataset(args, all_img_id, all_answer_id, all_image_feature, contain_search_res)
    else:
        dataset = NELDataset(args, features, contain_search_res)
    return dataset




def train(args, train_dataset, nel_model, fold=""):
    tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                              collate_fn=train_collate_fn if args.dataset not in ["person", "diverse"] else None)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]

    params_decay = [p for n, p in nel_model.named_parameters() if not any(nd in n for nd in no_decay)]
    params_nodecay = [p for n, p in nel_model.named_parameters() if any(nd in n for nd in no_decay)]

    optimizer_grouped_parameters = [{"params": params_decay, "weight_decay": args.weight_decay},
                                    {"params": params_nodecay, "weight_decay": 0.0}, ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon,
                      no_deprecation_warning=True)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        logger.info("loading the existing model weights")
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))


    # shareSwin_model = SwinForAffwildClassification(args).to(args.device)
    # '''loading the pretrained SWIN (Swin Transformer) model and updating the parameter dictionary.'''
    # model_dict = shareSwin_model.state_dict()
    # pretrained_dict = torch.load(args.pretrained_backbone_path)['state_dict']
    # new_pretrained_dict = {}
    # for k in model_dict:
    #     if k in pretrained_dict:
    #         if k == 'classifier.weight':
    #             continue
    #         if k == 'classifier.bias':
    #             continue
    #         if k[:5] == 'swin.':
    #             k_val = k[5:]
    #         else:
    #             k_val = k
    #         new_pretrained_dict[k] = pretrained_dict['backbone.' + k_val] # tradition training
    # model_dict.update(new_pretrained_dict)
    # shareSwin_model.load_state_dict(model_dict)
    
    

    global_step, epochs_trained, steps_trained_in_current_epoch = 0, 0, 0
    best_result = [0, 0, 0, 0]


    tr_loss, logging_loss = 0.0, 0.0
    nel_model.zero_grad()

    set_seed(args)  # Added here for reproductibility
    

    epoch_start_time = time()
    step_start_time = None
    for epoch in range(epochs_trained, int(args.num_train_epochs)):
        if epoch == epochs_trained:
            print(f"Epoch: {epoch + 1}/{int(args.num_train_epochs)} begin.")
        else:
            print(f"Epoch: {epoch + 1}/{int(args.num_train_epochs)} begin ({(time() - epoch_start_time) / (epoch - epochs_trained):2f}s/epoch).")
        epoch_iterator = train_dataloader
        num_steps = len(train_dataloader)
        for step, batch in tqdm(enumerate(epoch_iterator), desc="Train", ncols=50, total=num_steps):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            nel_model.train()

            for k in batch:
                batch[k] = batch[k].to(args.device) if isinstance(batch[k], torch.Tensor) else batch[k]
                    
            nel_inputs = {
                "mention": batch["mention_feature"].float(),
                "text": batch["text_feature"].float(),
                "total": batch["total_feature"].float(),
                "profile": batch["profile_feature"].float(),
                "segement": batch["segement_feature"].float(),
                "pos_feats": batch["pos"],
                "neg_feats": batch["neg"],
                "identity": batch["identity_feature"].float()
            }

            outputs = nel_model(**nel_inputs)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(nel_model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                nel_model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # # Log metrics
                    tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                    if step_start_time is None:
                        step_start_time = time()
                        print()
                        logger.info(
                            f"loss_{global_step}: {(tr_loss - logging_loss) / args.logging_steps}, epoch {epoch + 1}: {step + 1}/{num_steps}")
                    else:
                        print()
                        logger.info(
                            f"epoch {epoch + 1}, loss: {(tr_loss - logging_loss) / args.logging_steps}")
                        step_start_time = time()
                    logging_loss = tr_loss

                # save model if args.save_steps>0
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results, _ = evaluate(args, nel_model, mode=f"dev{fold}")[:2]
                        show_result = list(results.values())
                        if show_result[0] > best_result[0]:
                            best_result = show_result
                            best_result.append(epoch)
                        logger.info(
                            "### EVAL RESULT: {0:.3f}, {1:.3f}, {2:.3f}, {3:.3f} at {4}".format(show_result[0]*100,
                                                                                                show_result[1]*100,
                                                                                                show_result[2]*100,
                                                                                                show_result[3]*100, epoch))
                        logger.info(
                            "### BEST RESULT: {0:.3f}, {1:.3f}, {2:.3f}, {3:.3f} at {4}".format(best_result[0]*100,
                                                                                                best_result[1]*100,
                                                                                                best_result[2]*100,
                                                                                                best_result[3]*100,
                                                                                                best_result[4]))

                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)

                        tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                        tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)

            if 0 < args.max_steps < global_step:
                break

        if 0 < args.max_steps < global_step:
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()
 


def evaluate(args, nel_model, mode):
    time_eval_beg = time()

    eval_dataset = load_and_cache_examples(args, mode, args.dataset, logger)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    if args.dataset in ["person", "diverse"]:
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.train_batch_size)
    else:
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=train_collate_fn)

    eval_loss, nb_eval_steps = 0.0, 0
    nel_model.eval()


    # shareSwin_model = SwinForAffwildClassification(args).to(args.device)
    # '''loading the pretrained SWIN (Swin Transformer) model and updating the parameter dictionary.'''
    # model_dict = shareSwin_model.state_dict()
    # pretrained_dict = torch.load(args.pretrained_backbone_path)['state_dict']
    # new_pretrained_dict = {}
    # for k in model_dict:
    #     if k in pretrained_dict:
    #         if k == 'classifier.weight':
    #             continue
    #         if k == 'classifier.bias':
    #             continue
    #         if k[:5] == 'swin.':
    #             k_val = k[5:]
    #         else:
    #             k_val = k
    #         new_pretrained_dict[k] = pretrained_dict['backbone.' + k_val] # tradition training
    # model_dict.update(new_pretrained_dict)
    # shareSwin_model.load_state_dict(model_dict)
    # shareSwin_model.eval()


    all_ranks = []
    time_eval_rcd = time()
    nsteps = len(eval_dataloader)
    with torch.no_grad():
        for i, batch in tqdm(enumerate(eval_dataloader), desc='Eval', ncols=50, total=nsteps):
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(args.device)

            nel_inputs = {
                "mention": batch["mention_feature"].float(),
                "text": batch["text_feature"].float(),
                "total": batch["total_feature"].float(),
                "profile": batch["profile_feature"].float(),
                "segement": batch["segement_feature"].float(),
                "pos_feats": batch["pos"],
                "neg_feats": batch["neg"],
                "identity": batch["identity_feature"].float()
            }

            outputs = nel_model(**nel_inputs)
            tmp_eval_loss, query = outputs[:2]

            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating

            tmp_eval_loss = tmp_eval_loss.mean()
            eval_loss += tmp_eval_loss

            pos_feat_trans = batch["pos"]
            neg_feat_trans = batch["search_res"]

            rank_list, sim_p, sim_n = cal_top_k(args, query, pos_feat_trans, neg_feat_trans)

            all_ranks.extend(rank_list)

            nb_eval_steps += 1

            if (i + 1) % 100 == 0:
                print(f"{mode}: {i + 1}/{nsteps}, loss: {tmp_eval_loss}, {time() - time_eval_rcd:.2f}s/100steps")
                time_eval_rcd = time()
    eval_loss = eval_loss.item() / nb_eval_steps
    all_ranks = np.array(all_ranks)
    results = {
        "top1": int(sum(all_ranks <= 1)) / len(eval_dataset),
        "top5": int(sum(all_ranks <= 5)) / len(eval_dataset),
        "top10": int(sum(all_ranks <= 10)) / len(eval_dataset),
        "top20": int(sum(all_ranks <= 20)) / len(eval_dataset),
    }

    logger.info(f"Eval loss: {eval_loss}, Eval time: {time() - time_eval_beg:2f}")

    return results, eval_loss, all_ranks




def main():
    args = parse_arg()
    args.n_gpu = 0
    args.device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    set_seed(args)

    nel_model = NELModel(args)
    for p in nel_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    print(args.device)
    nel_model.to(args.device)
    train_dataset = load_and_cache_examples(args, "train", args.dataset, logger)
    train(args, train_dataset, nel_model)


if __name__ == "__main__":
    main()