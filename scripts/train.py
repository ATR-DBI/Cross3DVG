import os
import sys
import json
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import subprocess
from collections import defaultdict
from torch.utils.data import DataLoader
from datetime import datetime
from copy import deepcopy
from socket import gethostname

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder

from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.dataset import ReferenceDataset 
from lib.solver import Solver
from models.refnet import TransReferNet
from models.scanrefer.refnet import ScanReferNet
from scripts.utils.AdamW import AdamW
from scripts.utils.script_utils import set_params_lr_dict

print(sys.path, '<< sys path')


def get_commit_hash():
    #cmd = "git rev-parse --short HEAD"
    cmd = "git rev-parse HEAD"
    hash = subprocess.check_output(cmd.split()).strip().decode('utf-8')
    return hash


def get_dataloader(args, DC, CONF, scanrefer, scanrefer_new, all_scene_list, split, augment, shuffle=True):
    dataset = ReferenceDataset(
        args=args,
        DC=DC,
        CONF=CONF,
        scanrefer=scanrefer[split],
        scanrefer_new=scanrefer_new[split],
        scanrefer_all_scene=all_scene_list,
        split=split,
        num_points=args.num_points,
        use_height=args.use_height,
        use_color=args.use_color,
        use_normal=args.use_normal,
        lang_num_max=args.lang_num_max,
        augment=augment,
        shuffle=shuffle
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=4, pin_memory=True) 
    return dataset, dataloader

def get_model(args, DC, CONF):
    # initiate model
    input_channels = int(args.use_normal) * 3 + int(args.use_color) * 3 + int(args.use_height)
    
    if args.tokenizer == 'clip':
        if 'ViT' in args.clip_model:
            clip_size = 512
        elif 'RN50' == args.clip_model:
            clip_size = 1024
        elif 'RN50x4' == args.clip_model:
            clip_size = 640                
        else:
            raise NotImplemented                
        emb_size = clip_size
    elif args.tokenizer == 'bert':        
        from transformers import AutoConfig
        bert_config = AutoConfig.from_pretrained(args.bert_model)
        if hasattr(bert_config, "hidden_size"):
            emb_size = bert_config.hidden_size
        else:
            # for distllbert
            emb_size = bert_config.dim        
    else:
        emb_size = 300


    if args.model == 'transrefer':
        model = TransReferNet(
            args,
            num_class=DC.num_class,
            num_heading_bin=DC.num_heading_bin,
            num_size_cluster=DC.num_size_cluster,
            mean_size_arr=DC.mean_size_arr,
            input_feature_dim=input_channels,
            num_proposal=args.num_proposals,
            use_lang_classifier=(not args.no_lang_cls),
            use_bidir=args.use_bidir,
            no_reference=args.no_reference,
            emb_size=emb_size,
            dataset_config=DC
        )
    elif args.model == 'scanrefer':
        model = ScanReferNet(
            num_class=DC.num_class,
            num_heading_bin=DC.num_heading_bin,
            num_size_cluster=DC.num_size_cluster,
            mean_size_arr=DC.mean_size_arr,
            input_feature_dim=input_channels,
            num_proposal=args.num_proposals,
            use_lang_classifier=(not args.no_lang_cls),
            use_bidir=args.use_bidir,
            no_reference=args.no_reference,
            use_match_gru=args.use_match_gru,
        )       
    else:
        raise NotImplementedError

    # trainable model
    if args.use_pretrained:
        # load model
        if args.model == 'transrefer':
            print("loading pretrained VoteNet...")
            pretrained_model = TransReferNet(
                args,
                num_class=DC.num_class,
                num_heading_bin=DC.num_heading_bin,
                num_size_cluster=DC.num_size_cluster,
                mean_size_arr=DC.mean_size_arr,
                num_proposal=args.num_proposals,
                input_feature_dim=input_channels,
                use_bidir=args.use_bidir,
                no_reference=True,
                dataset_config=DC
            )
        elif args.model == 'scanrefer':
            pretrained_model = ScanReferNet(
                num_class=DC.num_class,
                num_heading_bin=DC.num_heading_bin,
                num_size_cluster=DC.num_size_cluster,
                mean_size_arr=DC.mean_size_arr,
                num_proposal=args.num_proposals,
                input_feature_dim=input_channels,
                use_bidir=args.use_bidir,
                no_reference=True,
                use_match_gru=args.use_match_gru,
            )    
        else:
            raise NotImplementedError

        pretrained_path = os.path.join(CONF.PATH.OUTPUT, args.use_pretrained, "model_last.pth")
        pretrained_model.load_state_dict(torch.load(pretrained_path), strict=False)

        # mount
        model.backbone_net = pretrained_model.backbone_net
        model.vgen = pretrained_model.vgen
        model.proposal = pretrained_model.proposal

        if args.no_detection:
            # freeze pointnet++ backbone
            for param in model.backbone_net.parameters():
                param.requires_grad = False

            # freeze voting
            for param in model.vgen.parameters():
                param.requires_grad = False

            # freeze detector
            for param in model.proposal.parameters():
                param.requires_grad = False
    # to CUDA
    model = model.cuda()

    return model


def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params

def get_solver(args, DC, CONF, dataloader):
    
    model = get_model(args, DC, CONF)
    # different lr for various modules.
    weight_dict = {
        'detr': {'lr': 0.0001},
        'lang': {'lr': 0.0005},
        'match': {'lr': 0.0005},
    }
    params = set_params_lr_dict(model, base_lr=args.lr, weight_decay=args.wd, weight_dict=weight_dict)
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.wd, amsgrad=args.amsgrad)

    if args.use_checkpoint:
        print("loading checkpoint {}...".format(args.use_checkpoint))
        stamp = args.use_checkpoint
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        checkpoint = torch.load(os.path.join(CONF.PATH.OUTPUT, args.use_checkpoint, "checkpoint.tar"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.tag: stamp += "_"+args.tag.upper()
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        os.makedirs(root, exist_ok=True)

    # scheduler parameters for training solely the detection pipeline
    LR_DECAY_STEP = [80, 120, 160] if args.no_reference else None
    if args.coslr:
        LR_DECAY_STEP = {
            'type': 'cosine',
            'T_max': args.epoch,
            'eta_min': 1e-5,
        }
    LR_DECAY_RATE = 0.1 if args.no_reference else None
    BN_DECAY_STEP = 20 if args.no_reference else None
    BN_DECAY_RATE = 0.5 if args.no_reference else None

    print('LR&BN_DECAY', LR_DECAY_STEP, LR_DECAY_RATE, BN_DECAY_STEP, BN_DECAY_RATE, flush=True)
    solver = Solver(
        args,
        model=model,
        DC=DC,
        CONF=CONF,
        dataloader=dataloader,
        optimizer=optimizer,
        stamp=stamp,
        val_step=args.val_step,
        detection=not args.no_detection,
        reference=not args.no_reference,
        use_lang_classifier=not args.no_lang_cls,
        lr_decay_step=LR_DECAY_STEP,
        lr_decay_rate=LR_DECAY_RATE,
        bn_decay_step=BN_DECAY_STEP,
        bn_decay_rate=BN_DECAY_RATE
    )
    num_params = get_num_params(model)

    return solver, num_params, root


def save_info(args, root, num_params, train_dataset, val_dataset):
    info = {}
    for key, value in vars(args).items():
        info[key] = value

    info["num_train"] = len(train_dataset)
    info["num_val"] = len(val_dataset)
    info["num_train_scenes"] = len(train_dataset.scene_list)
    info["num_val_scenes"] = len(val_dataset.scene_list)
    info["num_params"] = num_params
    
    info["git_commit_hash"] = get_commit_hash()
    info["hostname"] = gethostname()

    with open(os.path.join(root, "info.json"), "w") as f:
        json.dump(info, f, indent=4)
        
    # save commandline 
    cmd = " ".join([v for v in sys.argv])
    cmd_file = os.path.join(root, "cmdline.txt")
    open(cmd_file, 'w').write(cmd)        


def get_scannet_scene_list(split):
    scene_list = sorted(
        [line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])
    return scene_list


def get_scanrefer(scanrefer_train, scanrefer_val, num_scenes, lang_num_max):
    scanrefer_train = sorted(scanrefer_train, key=lambda x: x['scene_id'])
    scanrefer_val = sorted(scanrefer_val, key=lambda x: x['scene_id'])

    if args.no_reference:
        raise NotImplemented 
        '''
        train_scene_list = get_scannet_scene_list("train")
        new_scanrefer_train = []
        for scene_id in train_scene_list:
            data = deepcopy(scanrefer_train[0])
            data["scene_id"] = scene_id
            new_scanrefer_train.append(data)

        val_scene_list = get_scannet_scene_list("val")
        new_scanrefer_val = []
        for scene_id in val_scene_list:
            data = deepcopy(scanrefer_val[0])
            data["scene_id"] = scene_id
            new_scanrefer_val.append(data)
        '''
    else:
        # get initial scene list
        train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))
        val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_val])))
        
        MAX_NUM_OBJ = 128
        train_object_set = defaultdict(set)
        for refer in scanrefer_train:
            train_object_set[refer['scene_id']].add(refer['object_id'])
            
        val_object_set = defaultdict(set)
        for refer in scanrefer_val:
            val_object_set[refer['scene_id']].add(refer['object_id'])    

        train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train if len(train_object_set[data["scene_id"]]) <= MAX_NUM_OBJ])))
        val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_val if len(val_object_set[data["scene_id"]]) <= MAX_NUM_OBJ])))
        
        if num_scenes == -1:
            num_scenes = len(train_scene_list)
        else:
            assert len(train_scene_list) >= num_scenes

        # slice train_scene_list
        train_scene_list = train_scene_list[:num_scenes]
        # filter data in chosen scenes
        new_scanrefer_train = []
        scanrefer_train_new = []
        scanrefer_train_new_scene = []
        scene_id = ""
        
        for data in scanrefer_train:
            if data["scene_id"] in train_scene_list:
                new_scanrefer_train.append(data)
                if scene_id != data["scene_id"]:
                    scene_id = data["scene_id"]
                    if len(scanrefer_train_new_scene) > 0:
                        scanrefer_train_new.append(scanrefer_train_new_scene)
                    scanrefer_train_new_scene = []
                if len(scanrefer_train_new_scene) >= lang_num_max:
                    scanrefer_train_new.append(scanrefer_train_new_scene)
                    scanrefer_train_new_scene = []
                scanrefer_train_new_scene.append(data)
        scanrefer_train_new.append(scanrefer_train_new_scene)

        new_scanrefer_val = scanrefer_val
        scanrefer_val_new = []
        scanrefer_val_new_scene = []
        scene_id = ""
        for data in scanrefer_val:
            if scene_id != data["scene_id"]:
                scene_id = data["scene_id"]
                if len(scanrefer_val_new_scene) > 0:
                    scanrefer_val_new.append(scanrefer_val_new_scene)
                scanrefer_val_new_scene = []
            if len(scanrefer_val_new_scene) >= lang_num_max:
                scanrefer_val_new.append(scanrefer_val_new_scene)
                scanrefer_val_new_scene = []
            scanrefer_val_new_scene.append(data)
        scanrefer_val_new.append(scanrefer_val_new_scene)

    sum = 0
    for i in range(len(scanrefer_train_new)):
        sum += len(scanrefer_train_new[i])
    print("training sample numbers", sum)
    # all scanrefer scene
    all_scene_list = train_scene_list + val_scene_list
    print("train on {} samples and val on {} samples".format(len(new_scanrefer_train), len(new_scanrefer_val))) 
    return new_scanrefer_train, new_scanrefer_val, all_scene_list, scanrefer_train_new, scanrefer_val_new


def train(args, DC):
    # init training dataset
    print("preparing data...")
    
    if args.dataset == "scanrefer":
        from lib.scanrefer_config import CONF
        SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "scanrefer/meta_data", "ScanRefer_filtered_train.json")))
        SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "scanrefer/meta_data", "ScanRefer_filtered_val.json")))        
    elif "riorefer" in args.dataset:
        from lib.riorefer_config import CONF
        SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "riorefer/meta_data", "RIORefer_train.json")))
        SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "riorefer/meta_data", "RIORefer_val.json")))        
    else:
        raise NotImplementedError
    
    scanrefer_train, scanrefer_val, all_scene_list, scanrefer_train_new, scanrefer_val_new = get_scanrefer(
        SCANREFER_TRAIN, SCANREFER_VAL, args.num_scenes, args.lang_num_max)
    scanrefer = {
        "train": scanrefer_train,
        "val": scanrefer_val
    }
    scanrefer_new = {
        "train": scanrefer_train_new,
        "val": scanrefer_val_new
    }

    train_dataset, train_dataloader = get_dataloader(args, DC, CONF, scanrefer, scanrefer_new, all_scene_list, "train", augment=True)
    val_dataset, val_dataloader = get_dataloader(args, DC, CONF, scanrefer, scanrefer_new, all_scene_list, "val", augment=False)
    dataloader = {
        "train": train_dataloader,
        "val": val_dataloader
    }

    print("initializing...")
    solver, num_params, root = get_solver(args, DC, CONF, dataloader)

    print("Start training...\n")
    save_info(args, root, num_params, train_dataset, val_dataset)
    solver(args.epoch, args.verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset 
    parser.add_argument("--dataset", type=str, help="refer dataset", default="scanrefer") # scanrefer, riorefer
    parser.add_argument("--labelset", type=str, help="label set", default="head_common_tail") # head, head_common, head_common_tail
    # basic setting
    parser.add_argument("--tag", type=str, help="tag for the training, e.g. cuda_wl", default="")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--batch_size", type=int, help="batch size", default=8)
    parser.add_argument("--epoch", type=int, help="number of epochs", default=200)
    parser.add_argument("--verbose", type=int, help="iterations of showing verbose", default=10)
    parser.add_argument("--val_step", type=int, help="iterations of validating", default=5000)
    parser.add_argument("--lr", type=float, help="learning rate", default=0.002)
    parser.add_argument("--wd", type=float, help="weight decay", default=1e-5)
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--coslr", action='store_true', help="cosine learning rate")
    parser.add_argument("--amsgrad", action='store_true', help="optimizer with amsgrad")
    parser.add_argument("--use_pretrained", type=str,
                        help="Specify the folder name containing the pretrained detection module.")
    parser.add_argument("--use_checkpoint", type=str, help="Specify the checkpoint root", default="")
    #
    parser.add_argument("--lang_num_max", type=int, help="lang num max", default=32)
    parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
    parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
    parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
    parser.add_argument("--no_augment", action="store_true", help="Do NOT use augment on trainingset (not used)")
    parser.add_argument("--no_lang_cls", action="store_true", help="Do NOT use language classifier.")
    parser.add_argument("--no_detection", action="store_true", help="Do NOT train the detection module.")
    parser.add_argument("--no_reference", action="store_true", help="Do NOT train the localization module.")
    parser.add_argument("--use_height", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
    # tokenizer
    parser.add_argument("--tokenizer", type=str, help="Pretrained tokenizer name", default="glove") # or bpemb, bert
    parser.add_argument("--bpemb_dim", type=int, default=300, help="")
    parser.add_argument("--bpemb_vs", type=int, default=10000, help="") # 10000 50000
    parser.add_argument("--bert_model", type=str, help="bert model name", default="bert-base-uncased")     
    parser.add_argument("--freeze_bert", action="store_true", help="")    

    # model
    parser.add_argument("--model", type=str, help="", default="transrefer")     
    parser.add_argument("--hidden_size", type=int, help="", default=128)    
    parser.add_argument("--use_bidir", action="store_true", help="Use bi-directional GRU.")
    # model (scanrefer)
    parser.add_argument("--use_match_gru", action='store_true', help="use gru for objects' context modeling")
    # model (cross3dvg)
    parser.add_argument("--use_text_clip", action="store_true", help="")
    parser.add_argument("--use_image_clip", action="store_true", help="")
    parser.add_argument("--use_text_clip_norm", action="store_true", help="")    
    parser.add_argument("--use_image_clip_norm", action="store_true", help="")    
    ## use_image_clip
    parser.add_argument("--num_image_clip", type=int, default=10, help="")
    parser.add_argument("--image_clip_cossim", type=float, default=0.9, help="")    
    # backbone module
    parser.add_argument("--proposal_module_votenet", action="store_true", help="")
    ## deter
    parser.add_argument("--det_dec_layers", type=int, help="", default=2)    
    parser.add_argument("--det_hidden_dim", type=int, help="", default=288)    
    parser.add_argument("--det_dim_feedforward", type=int, help="", default=2048)    
    ## pointnet
    parser.add_argument("--proposal_size", type=int, help="", default=128)        
    # match module
    parser.add_argument("--use_cache", action="store_true", help="")
    parser.add_argument("--clip_model", type=str, default="ViT-B32") # ViT-B32 RN50x4
    parser.add_argument("--caption_type", type=str, default="obj_desc") # obj_name, obj_desc, obj_name_desc
    parser.add_argument("--match_joint_score", action="store_true", help="")
    # loss weight
    parser.add_argument("--vote_weight", type=float, default=1.0, help="")
    parser.add_argument("--objectness_weight", type=float, default=0.1, help="")
    parser.add_argument("--box_weight", type=float, default=1.0, help="")
    parser.add_argument("--sem_cls_weight", type=float, default=0.1, help="")
    parser.add_argument("--ref_weight", type=float, default=0.03, help="")
    parser.add_argument("--lang_weight", type=float, default=0.03, help="")
    parser.add_argument("--clip_weight", type=float, default=1.0, help="")
    parser.add_argument("--corr_weight", type=float, default=0.03, help="")
    args = parser.parse_args()

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    
    DC = ScannetDatasetConfig(args.labelset)
    
    train(args, DC)
