'''
File Created: Monday, 25th November 2019 1:35:30 pm
Author: Dave Zhenyu Chen (zhenyu.chen@tum.de)
'''
import os, re, sys
import time
import gc
import glob
import pandas as pd
import joblib
import json
import pickle
from collections import defaultdict
import numpy as np
import multiprocessing as mp
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from utils.pc_utils import random_sampling, rotx, roty, rotz
from data.scannet.model_util_scannet import rotate_aligned_boxes, rotate_aligned_boxes_along_axis, ScannetDatasetConfig
import random

MAX_NUM_OBJ = 128
MAX_NUM_FRAME = 430 # >= 417
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])


def _load_data_by_scene(scene_id, SCAN_DATA):
    scene_data = {}
    scene_data["mesh_vertices"] = np.load(os.path.join(SCAN_DATA, scene_id)+"_aligned_vert.npy") # axis-aligned
    scene_data["instance_labels"] = np.load(os.path.join(SCAN_DATA, scene_id)+"_ins_label.npy")
    scene_data["semantic_labels"] = np.load(os.path.join(SCAN_DATA, scene_id)+"_sem_label.npy")
    scene_data["instance_bboxes"] = np.load(os.path.join(SCAN_DATA, scene_id)+"_aligned_bbox.npy")
    return (scene_id, scene_data)


class ReferenceDataset(Dataset):
    
    def __init__(self, args, DC, CONF, scanrefer, scanrefer_new, scanrefer_all_scene,
        split="train",
        num_points=40000,
        lang_num_max=32,
        use_height=False,
        use_color=False,
        use_normal=False,
        augment=False,
        shuffle=False):
        self.args = args
        self.DC = DC        
        self.CONF = CONF
        
        # data path
        self.glove_pickle_file = os.path.join(self.CONF.PATH.DATA, "glove.p")        

        self.scanrefer = scanrefer
        self.scanrefer_new = scanrefer_new
        self.scanrefer_new_len = len(scanrefer_new)
        self.scanrefer_all_scene = scanrefer_all_scene # all scene_ids in scanrefer
        self.split = split
        self.num_points = num_points
        self.use_color = use_color
        self.use_height = use_height
        self.use_normal = use_normal
        self.augment = augment
        self.lang_num_max = lang_num_max
        
        norm_mean= [0.5, 0.5, 0.5]
        norm_std = [0.5, 0.5, 0.5]
        train_transform = [
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]
        val_transform = [
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]
        self.train_transform = transforms.Compose(train_transform)
        self.val_transform = transforms.Compose(val_transform)

        if self.args.use_cache:
            self.cached_data = {}
            self.cached_data['frames'] = {}
            
        if self.args.dataset == 'scanrefer':
            clip_ref = 'scanrefer_clip'
            clip_obj = 'scannet_clip'
        elif self.args.dataset == 'riorefer':
            clip_ref = 'riorefer_clip'
            clip_obj = 'rio_clip'
        else:
            raise NotImplementedError
        
        if 'ViT' in self.args.clip_model:
            self.clip_size = 512
        elif 'RN50' == self.args.clip_model:
            self.clip_size = 1024
        elif 'RN50x4' == self.args.clip_model:
            self.clip_size = 640                
        else:
            raise NotImplemented        

        caption_type = '.'+self.args.caption_type
        clip_model = '.'+self.args.clip_model
        use_text_norm = ('.norm' if self.args.use_text_clip_norm else '')
        use_image_norm = ('.norm' if self.args.use_image_clip_norm else '')

        if self.args.use_text_clip:
            gc.disable()
            # get text_feature_objcap by scene_id, object_id, ann_id
            with open(os.path.join(self.CONF.PATH.TEXT_FEATURE, clip_ref + caption_type + clip_model + use_text_norm +'.pkl'), 'rb') as f:
                self.text_feature_objcap = pickle.load(f) 
            # get text_feature_objname by object_name
            with open(os.path.join(self.CONF.PATH.TEXT_FEATURE, clip_obj + clip_model + use_text_norm +'.pkl'), 'rb') as f:
                self.text_feature_objname = pickle.load(f)
            gc.enable()                
            # object_id -> object_name
            self.object_label_map = self.get_object_label_map()            
                        
        # CLIP
        if self.args.use_image_clip:
            # get image_feature by scene_id, frame_id (pose_id)
            with open(os.path.join(self.CONF.PATH.IMAGE_FEATURE , clip_ref + clip_model + use_image_norm +'.pkl'), 'rb') as f:
                self.image_feature = pickle.load(f)
                
        # load data
        self._load_data()
        self.multiview_data = {}
        # self.shuffled = False
        self.should_shuffle = shuffle
        # self.shuffle_data()                
        
    def __len__(self):
        return self.scanrefer_new_len
    
    def get_object_label_map(self):
        # get object_name by scene_id, object_id
        if self.args.dataset == 'scanrefer':
            map_dic = {}
            for scene_dir in glob.glob(os.path.join(self.CONF.PATH.SCANS, '*')):
                scene_id = os.path.basename(scene_dir)              
                agg_file = os.path.join(scene_dir, scene_id +'.aggregation.json')
                if not os.path.exists(agg_file):
                    continue
                for seg in json.load(open(agg_file))['segGroups']:
                    if scene_id not in map_dic:
                        map_dic[scene_id] = {}
                    map_dic[scene_id][seg['objectId']] = seg['label']
            return map_dic
        elif 'riorefer' in self.args.dataset:        
            map_dic = {}
            for scene_dir in glob.glob(os.path.join(self.CONF.PATH.SCANS, '*')):
                scene_id = os.path.basename(scene_dir)              
                agg_file = os.path.join(scene_dir, 'semseg.v2.json')
                if not os.path.exists(agg_file):
                    continue
                for seg in json.load(open(agg_file))['segGroups']:
                    if scene_id not in map_dic:
                        map_dic[scene_id] = {}
                    map_dic[scene_id][seg['objectId']] = seg['label']
            return map_dic
        else:
            raise NotImplementedError

    def split_scene_new(self,  scanrefer_data):
        scanrefer_train_new = []
        scanrefer_train_new_scene, scanrefer_train_scene = [], []
        scene_id = ''
        lang_num_max = self.lang_num_max
        for data in scanrefer_data:
            if scene_id != data["scene_id"]:
                scene_id = data["scene_id"]
                if len(scanrefer_train_scene) > 0:
                    if self.should_shuffle:
                        random.shuffle(scanrefer_train_scene)
                    for new_data in scanrefer_train_scene:
                        if len(scanrefer_train_new_scene) >= lang_num_max:
                            scanrefer_train_new.append(scanrefer_train_new_scene)
                            scanrefer_train_new_scene = []
                        scanrefer_train_new_scene.append(new_data)
                    if len(scanrefer_train_new_scene) > 0:
                        scanrefer_train_new.append(scanrefer_train_new_scene)
                        scanrefer_train_new_scene = []
                    scanrefer_train_scene = []
            scanrefer_train_scene.append(data)
        if len(scanrefer_train_scene) > 0:
            if self.should_shuffle:
                random.shuffle(scanrefer_train_scene)
            for new_data in scanrefer_train_scene:
                if len(scanrefer_train_new_scene) >= lang_num_max:
                    scanrefer_train_new.append(scanrefer_train_new_scene)
                    scanrefer_train_new_scene = []
                scanrefer_train_new_scene.append(new_data)
            if len(scanrefer_train_new_scene) > 0:
                scanrefer_train_new.append(scanrefer_train_new_scene)
                scanrefer_train_new_scene = []
        return scanrefer_train_new


    def shuffle_data(self):
        print('shuffle dataset data(lang)', flush=True)
        self.scanrefer_new = self.split_scene_new(self.scanrefer)
        if self.should_shuffle:
            random.shuffle(self.scanrefer_new)
        assert len(self.scanrefer_new) == self.scanrefer_new_len, 'assert scanrefer length right'
        print('shuffle done', flush=True)

    
    def __getitem__(self, idx):
        start = time.time()
        lang_num = len(self.scanrefer_new[idx])
        scene_id = self.scanrefer_new[idx][0]["scene_id"]

        object_id_list = []
        object_name_list = []
        ann_id_list = []
        
        if self.args.tokenizer == 'bert':
            lang_input_ids_list = []
            lang_token_type_ids_list = []
            lang_attention_mask_list = []
            lang_len_list = []

            for i in range(self.lang_num_max):
                if i < lang_num:
                    object_id = int(self.scanrefer_new[idx][i]["object_id"])
                    object_name = " ".join(self.scanrefer_new[idx][i]["object_name"].split("_"))
                    ann_id = self.scanrefer_new[idx][i]["ann_id"]
                    lang_input_ids = self.lang[scene_id][str(object_id)][ann_id]['input_ids']
                    lang_token_type_ids = self.lang[scene_id][str(object_id)][ann_id].get('token_type_ids', None)
                    lang_attention_mask = self.lang[scene_id][str(object_id)][ann_id]['attention_mask']
                    lang_len = self.lang_len[scene_id][str(object_id)][ann_id]
                    lang_len = lang_len if lang_len <= self.CONF.TRAIN.MAX_DES_LEN else self.CONF.TRAIN.MAX_DES_LEN

                object_id_list.append(object_id)
                object_name_list.append(object_name)
                ann_id_list.append(ann_id)
                lang_input_ids_list.append(lang_input_ids)
                if lang_token_type_ids is not None:
                    lang_token_type_ids_list.append(lang_token_type_ids)
                lang_attention_mask_list.append(lang_attention_mask)
                lang_len_list.append(lang_len)
        else:            
            lang_feat_list = []
            lang_len_list = []
            main_lang_feat_list = []
            main_lang_len_list = []
            first_obj_list = []
            unk_list = []            
            for i in range(self.lang_num_max):
                if i < lang_num:
                    object_id = int(self.scanrefer_new[idx][i]["object_id"])
                    object_name = self.scanrefer_new[idx][i]["object_name"].replace("_", " ")
                    ann_id = self.scanrefer_new[idx][i]["ann_id"]

                    lang_feat = self.lang[scene_id][str(object_id)][ann_id]
                    lang_len = self.lang_len[scene_id][str(object_id)][ann_id]
                    #lang_len = len(self.scanrefer_new[idx][i]["token"])
                    lang_len = lang_len if lang_len <= self.CONF.TRAIN.MAX_DES_LEN else self.CONF.TRAIN.MAX_DES_LEN
                    main_lang_feat = self.lang_main[scene_id][str(object_id)][ann_id]["main"]
                    main_lang_len = self.lang_main[scene_id][str(object_id)][ann_id]["len"]
                    first_obj = self.lang_main[scene_id][str(object_id)][ann_id]["first_obj"]
                    unk = self.lang_main[scene_id][str(object_id)][ann_id]["unk"]

                object_id_list.append(object_id)
                object_name_list.append(object_name)
                ann_id_list.append(ann_id)

                lang_feat_list.append(lang_feat)
                lang_len_list.append(lang_len)
                main_lang_feat_list.append(main_lang_feat)
                main_lang_len_list.append(main_lang_len)
                first_obj_list.append(first_obj)
                unk_list.append(unk)
                
        # get pc
        mesh_vertices = self.scene_data[scene_id]["mesh_vertices"]
        instance_labels = self.scene_data[scene_id]["instance_labels"]
        semantic_labels = self.scene_data[scene_id]["semantic_labels"]
        instance_bboxes = self.scene_data[scene_id]["instance_bboxes"]
        
        if not self.use_color:
            point_cloud = mesh_vertices[:,0:3] # do not use color for now
            pcl_color = mesh_vertices[:,3:6]
        else:
            point_cloud = mesh_vertices[:,0:6]
            point_cloud[:,3:6] = (point_cloud[:,3:6]-MEAN_COLOR_RGB)/256.0
            pcl_color = point_cloud[:,3:6]

        if self.use_normal:
            normals = mesh_vertices[:,6:9]
            point_cloud = np.concatenate([point_cloud, normals],1)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1)

        point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]
        pcl_color = pcl_color[choices]

        # ------------------------------- LABELS ------------------------------
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))

        ref_box_label_list = []
        ref_center_label_list = []
        ref_heading_class_label_list = []
        ref_heading_residual_label_list = []
        ref_size_class_label_list = []
        ref_size_residual_label_list = []
        ref_sem_label_list = []
        ref_text_feat_list = []

        if self.split != "test":
            num_bbox = instance_bboxes.shape[0] if instance_bboxes.shape[0] < MAX_NUM_OBJ else MAX_NUM_OBJ
            target_bboxes_mask[0:num_bbox] = 1
            target_bboxes[0:num_bbox,:] = instance_bboxes[:MAX_NUM_OBJ,0:6]

            point_votes = np.zeros([self.num_points, 3])
            point_votes_mask = np.zeros(self.num_points)

            # ------------------------------- DATA AUGMENTATION ------------------------------
            if self.augment:  # and not self.debug: # shape not changed; TODO scale
                if np.random.random() > 0.7:
                    # Flipping along the YZ plane
                    point_cloud[:, 0] = -1 * point_cloud[:, 0]
                    target_bboxes[:, 0] = -1 * target_bboxes[:, 0]

                if np.random.random() > 0.7:
                    # Flipping along the XZ plane
                    point_cloud[:, 1] = -1 * point_cloud[:, 1]
                    target_bboxes[:, 1] = -1 * target_bboxes[:, 1]

                    # Rotation along X-axis
                rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
                rot_mat = rotx(rot_angle)
                point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
                target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "x")

                # Rotation along Y-axis
                rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
                rot_mat = roty(rot_angle)
                point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
                target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "y")

                # Rotation along up-axis/Z-axis
                rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
                rot_mat = rotz(rot_angle)
                point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
                target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "z")

                # print('Warning! Dont Use Extra Augmentation!(votenet didnot use it)', flush=True)
                # NEW: scale from 0.8 to 1.2
                # print(rot_mat.shape, point_cloud.shape, flush=True)
                scale = np.random.uniform(-0.1, 0.1, (3, 3))
                scale = np.exp(scale)
                # print(scale, '<<< scale', flush=True)
                scale = scale * np.eye(3)
                point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], scale)
                if self.use_height:
                    point_cloud[:, 3] = point_cloud[:, 3] * float(scale[2, 2])
                target_bboxes[:, 0:3] = np.dot(target_bboxes[:, 0:3], scale)
                target_bboxes[:, 3:6] = np.dot(target_bboxes[:, 3:6], scale)

                # Translation
                point_cloud, target_bboxes = self._translate(point_cloud, target_bboxes)

            # compute votes *AFTER* augmentation
            # generate votes
            # Note: since there's no map between bbox instance labels and
            # pc instance_labels (it had been filtered
            # in the data preparation step) we'll compute the instance bbox
            # from the points sharing the same instance label.
            for i_instance in np.unique(instance_labels):
                # find all points belong to that instance
                ind = np.where(instance_labels == i_instance)[0]
                # find the semantic label
                if semantic_labels[ind[0]] in self.DC.label_ids:
                    x = point_cloud[ind,:3]
                    center = 0.5*(x.min(0) + x.max(0))
                    point_votes[ind, :] = center - x
                    point_votes_mask[ind] = 1.0
            point_votes = np.tile(point_votes, (1, 3)) # make 3 votes identical
            
            class_ind = [self.DC.label_id2class[int(x)] for x in instance_bboxes[:num_bbox,-2]]
            size_classes[0:num_bbox] = class_ind
            size_residuals[0:num_bbox, :] = target_bboxes[0:num_bbox, 3:6] - self.DC.mean_size_arr[class_ind,:]
            
            object_ref_feature = np.zeros([MAX_NUM_OBJ, self.clip_size])
            object_name_feature = np.zeros([MAX_NUM_OBJ, self.clip_size]) 
            if self.args.use_text_clip:
                for i, gt_id in enumerate(instance_bboxes[:num_bbox,-1]):
                    if gt_id in self.text_feature_objcap[scene_id]: # scene_id, object_id, ann_id
                        _object_ref_feature = random.choice(list(self.text_feature_objcap[scene_id][gt_id].values()))
                        # _object_ref_feature: clip_size
                        object_ref_feature[i] = _object_ref_feature
                    else:
                        object_name = self.object_label_map[scene_id][gt_id]
                        object_ref_feature[i] = self.text_feature_objname[object_name] # clip image feature of object_name
                        
                    object_name = self.object_label_map[scene_id][gt_id]                   
                    object_name_feature[i] = self.text_feature_objname[object_name] # clip text feature of object_name

            # construct the reference target label for each bbox
            for j in range(self.lang_num_max):
                ref_box_label = np.zeros(MAX_NUM_OBJ)

                for i, gt_id in enumerate(instance_bboxes[:num_bbox, -1]):
                    if gt_id == object_id_list[j]:
                        ref_box_label[i] = 1
                        ref_center_label = target_bboxes[i, 0:3]
                        ref_heading_class_label = angle_classes[i]
                        ref_heading_residual_label = angle_residuals[i]
                        ref_size_class_label = size_classes[i]
                        ref_size_residual_label = size_residuals[i]
                        ref_box_label_list.append(ref_box_label)
                        ref_center_label_list.append(ref_center_label)
                        ref_heading_class_label_list.append(ref_heading_class_label)
                        ref_heading_residual_label_list.append(ref_heading_residual_label)
                        ref_size_class_label_list.append(ref_size_class_label)
                        ref_size_residual_label_list.append(ref_size_residual_label)
                        ref_sem_label = instance_bboxes[i, -2]
                        ref_sem_label_list.append(ref_sem_label)
                        
                        if self.args.use_text_clip:
                            # ref_text_feat: clip_size (512)
                            ref_text_feature = self.text_feature_objcap[scene_id][gt_id][int(ann_id_list[j])]
                            ref_text_feat_list.append(ref_text_feature)
        else:
            num_bbox = 1
            point_votes = np.zeros([self.num_points, 9]) # make 3 votes identical
            point_votes_mask = np.zeros(self.num_points)
        
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        try:
            target_bboxes_semcls[0:num_bbox] = [self.DC.label_id2class[int(x)] for x in instance_bboxes[:,-2][0:num_bbox]]
        except KeyError:
            pass

        object_cat_list = []
        for i in range(self.lang_num_max):
            object_cat = self.raw2label[object_name_list[i]] if object_name_list[i] in self.raw2label else self.DC.type2class['others']            
            object_cat_list.append(object_cat)
            
        istrain = 0
        if self.split == "train":
            istrain = 1

        data_dict = {}
        data_dict["point_clouds"] = point_cloud.astype(np.float32) # point cloud data including features
        data_dict["istrain"] = istrain
        data_dict["center_label"] = target_bboxes.astype(np.float32)[:,0:3] # (MAX_NUM_OBJ, 3) for GT box center XYZ
        data_dict["heading_class_label"] = angle_classes.astype(np.int64) # (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
        data_dict["heading_residual_label"] = angle_residuals.astype(np.float32) # (MAX_NUM_OBJ,)
        data_dict["size_class_label"] = size_classes.astype(np.int64) # (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
        data_dict["size_residual_label"] = size_residuals.astype(np.float32) # (MAX_NUM_OBJ, 3)
        data_dict["num_bbox"] = np.array(num_bbox).astype(np.int64)
        data_dict["sem_cls_label"] = target_bboxes_semcls.astype(np.int64) # (MAX_NUM_OBJ,) semantic class index
        data_dict["box_label_mask"] = target_bboxes_mask.astype(np.float32) # (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
        data_dict["vote_label"] = point_votes.astype(np.float32)
        data_dict["vote_label_mask"] = point_votes_mask.astype(np.int64)
        data_dict["scan_idx"] = np.array(idx).astype(np.int64)
        data_dict["pcl_color"] = pcl_color
        data_dict['object_ref_feat'] = object_ref_feature.astype(np.float16) # (MAX_NUM_OBJ, 512) clip text feature
        data_dict['object_name_feat'] = object_name_feature.astype(np.float16) # (MAX_NUM_OBJ, 512) clip text feature
        data_dict["load_time"] = time.time() - start
        data_dict["lang_num"] = np.array(lang_num).astype(np.int64)

        if self.args.tokenizer == 'bert':
            data_dict["lang_input_ids_list"] = np.array(lang_input_ids_list).astype(np.int64)  # language feature vectors
            if len(lang_token_type_ids_list) > 0:
                data_dict["lang_token_type_ids_list"] = np.array(lang_token_type_ids_list).astype(np.int64)  # language feature vectors
            data_dict["lang_attention_mask_list"] = np.array(lang_attention_mask_list).astype(np.float32)  # language feature vectors
            data_dict["lang_len_list"] = np.array(lang_len_list).astype(np.int64)  # length of each description
        else:
            data_dict["unk"] = unk.astype(np.float32)
            # lang_feat_list: lang_num_max, max_num_tokens, glove_size (32, 126, 300)
            data_dict["lang_feat_list"] = np.array(lang_feat_list).astype(np.float32)  # language feature vectors
            data_dict["lang_len_list"] = np.array(lang_len_list).astype(np.int64)  # length of each description
            data_dict["main_lang_feat_list"] = np.array(main_lang_feat_list).astype(np.float32)  # main language feature vectors
            data_dict["main_lang_len_list"] = np.array(main_lang_len_list).astype(np.int64)  # length of each main description
            data_dict["first_obj_list"] = np.array(first_obj_list).astype(np.int64)
            data_dict["unk_list"] = np.array(unk_list).astype(np.float32)
        
        data_dict["ref_box_label_list"] = np.array(ref_box_label_list).astype(np.int64)  # 0/1 reference labels for each object bbox
        data_dict["ref_center_label_list"] = np.array(ref_center_label_list).astype(np.float32)
        data_dict["ref_heading_class_label_list"] = np.array(ref_heading_class_label_list).astype(np.int64)
        data_dict["ref_heading_residual_label_list"] = np.array(ref_heading_residual_label_list).astype(np.int64)
        data_dict["ref_size_class_label_list"] = np.array(ref_size_class_label_list).astype(np.int64)
        data_dict["ref_size_residual_label_list"] = np.array(ref_size_residual_label_list).astype(np.float32)
        data_dict["ref_sem_label_list"] = np.array(ref_sem_label_list).astype(np.int64) 
        # lang_num_max, clip_size
        data_dict["ref_text_feat"] = np.array(ref_text_feat_list).astype(np.float32) 
        
        if self.args.use_image_clip:
            # image_feat: num_frame, 512
            image_feats = np.stack(list(self.image_feature[scene_id].values())) # ordered by frame_id
            num_frame = image_feats.shape[0]

            data_dict["image_feats"] = np.zeros((MAX_NUM_FRAME, self.clip_size))
            data_dict["image_feats"][:num_frame] = image_feats
            data_dict["image_feats"] = data_dict["image_feats"].astype(np.float32)
            frame_ids = np.stack(list(self.scene_data[scene_id]["camera_poses"].keys())) # num_pose
            data_dict["frame_id"] = np.zeros(MAX_NUM_FRAME)
            frame_ids = [int(str(x).replace('frame-', '')) for x in frame_ids]
            data_dict["frame_id"][:num_frame] = frame_ids         
            camera_poses = np.stack(list(self.scene_data[scene_id]["camera_poses"].values())) # num_frame, 4, 4
            data_dict["camera_poses"] = np.zeros((MAX_NUM_FRAME, 4, 4))
            data_dict["camera_poses"][:num_frame] = camera_poses
            
            assert image_feats.shape[0] == camera_poses.shape[0]
            # num_frame
            data_dict["num_frame"] = num_frame
            
        data_dict["object_id_list"] = np.array(object_id_list).astype(np.int64)
        data_dict["ann_id_list"] = np.array(ann_id_list).astype(np.int64)
        data_dict["object_cat_list"] = np.array(object_cat_list).astype(np.int64)
        
        unique_multiple_list = []
        for i in range(self.lang_num_max):
            object_id = object_id_list[i]
            ann_id = ann_id_list[i]
            unique_multiple = self.unique_multiple_lookup[scene_id][str(object_id)][ann_id]
            unique_multiple_list.append(unique_multiple)
        data_dict["unique_multiple_list"] = np.array(unique_multiple_list).astype(np.int64)

        return data_dict
    
    def _get_raw2label(self):
        # raw2label stores class_ind
        if self.args.dataset == 'scanrefer':
            label_mapping_file = os.path.join(self.CONF.PATH.META, "scannetv2-labels.combined.tsv")
            scannet_labels = self.DC.type2class.keys()
            scannet2label = {label: i for i, label in enumerate(scannet_labels)} 
            lines = [line.rstrip() for line in open(label_mapping_file)]
            lines = lines[1:]
            raw2label = {}
            for i in range(len(lines)):
                label_classes_set = set(scannet_labels)
                elements = lines[i].split('\t')
                raw_name = elements[1]
                if raw_name not in label_classes_set:
                    raw2label[raw_name] = scannet2label['others']
                else:
                    raw2label[raw_name] = scannet2label[raw_name]
        elif self.args.dataset == 'riorefer':
            label_mapping_file = os.path.join(self.CONF.PATH.META, "3RScan.v2_Semantic-Classes-Mapping.csv")            
            rio_labels = self.DC.type2class.keys()
            rio2label = {label: i for i, label in enumerate(rio_labels)}
            lines = [line.rstrip() for line in open(label_mapping_file)]
            lines = lines[3:]
            raw2label = {}
            for i in range(len(lines)):
                label_classes_set = set(rio_labels)
                elements = lines[i].split(',')
                raw_name = elements[1]
                if raw_name not in label_classes_set:
                    raw2label[raw_name] = rio2label['others']
                else:
                    raw2label[raw_name] = rio2label[raw_name]
        else:
            raise NotImplementedError
        return raw2label

    def _get_unique_multiple_lookup(self):
        all_sem_labels = {}
        cache = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))
            ann_id = data["ann_id"]

            if scene_id not in all_sem_labels:
                all_sem_labels[scene_id] = []
                
            if scene_id not in cache:
                cache[scene_id] = {}
                
            if object_id not in cache[scene_id]:
                cache[scene_id][object_id] = {}
                try:
                    all_sem_labels[scene_id].append(self.raw2label[object_name])
                except KeyError:
                    all_sem_labels[scene_id].append(17)

        # convert to numpy array
        all_sem_labels = {scene_id: np.array(all_sem_labels[scene_id]) for scene_id in all_sem_labels.keys()}

        unique_multiple_lookup = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))
            ann_id = data["ann_id"]

            try:
                sem_label = self.raw2label[object_name]
            except KeyError:
                sem_label = self.DC.type2class['others'] 

            unique_multiple = 0 if (all_sem_labels[scene_id] == sem_label).sum() == 1 else 1

            # store
            if scene_id not in unique_multiple_lookup:
                unique_multiple_lookup[scene_id] = {}

            if object_id not in unique_multiple_lookup[scene_id]:
                unique_multiple_lookup[scene_id][object_id] = {}

            if ann_id not in unique_multiple_lookup[scene_id][object_id]:
                unique_multiple_lookup[scene_id][object_id][ann_id] = None

            unique_multiple_lookup[scene_id][object_id][ann_id] = unique_multiple

        return unique_multiple_lookup


    def _tranform_glove_des(self):
        with open(self.glove_pickle_file, "rb") as f:
            glove = pickle.load(f)
        lang = {}
        lang_len = {}
        lang_main = {}
        scene_id_pre = ""
        i = 0
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            ann_id = data["ann_id"]
            object_name = " ".join(data["object_name"].split("_"))

            if scene_id not in lang:
                lang[scene_id] = {}
                lang_len[scene_id] = {}
                lang_main[scene_id] = {}

            if object_id not in lang[scene_id]:
                lang[scene_id][object_id] = {}
                lang_len[scene_id][object_id] = {}
                lang_main[scene_id][object_id] = {}

            if ann_id not in lang[scene_id][object_id]:
                lang[scene_id][object_id][ann_id] = {}
                lang_len[scene_id][object_id][ann_id] = {}
                lang_main[scene_id][object_id][ann_id] = {}
                lang_main[scene_id][object_id][ann_id]["main"] = {}
                lang_main[scene_id][object_id][ann_id]["len"] = 0
                lang_main[scene_id][object_id][ann_id]["first_obj"] = -1
                lang_main[scene_id][object_id][ann_id]["unk"] = glove["unk"]

            # tokenize the description
            tokens = data["token"]
            lang_len[scene_id][object_id][ann_id] = len(tokens)
            embeddings = np.zeros((self.CONF.TRAIN.MAX_DES_LEN, 300))
            main_embeddings = np.zeros((self.CONF.TRAIN.MAX_DES_LEN, 300))
            pd = 1

            main_object_cat = self.raw2label[object_name] if object_name in self.raw2label else self.DC.type2class['others']
            for token_id in range(self.CONF.TRAIN.MAX_DES_LEN):
                if token_id < len(tokens):
                    token = tokens[token_id]
                    if token in glove:
                        embeddings[token_id] = glove[token]
                    else:
                        embeddings[token_id] = glove["pad"]
                    if pd == 1:
                        if token in glove:
                            main_embeddings[token_id] = glove[token]
                        else:
                            main_embeddings[token_id] = glove["unk"]
                        if token == ".":
                            pd = 0
                            lang_main[scene_id][object_id][ann_id]["len"] = token_id + 1
                    object_cat = self.raw2label[token] if token in self.raw2label else -1
                    is_two_words = 0
                    if token_id + 1 < len(tokens):
                        token_new = token + " " + tokens[token_id+1]
                        object_cat_new = self.raw2label[token_new] if token_new in self.raw2label else -1
                        if object_cat_new != -1:
                            object_cat = object_cat_new
                            is_two_words = 1
                    if lang_main[scene_id][object_id][ann_id]["first_obj"] == -1 and object_cat == main_object_cat:
                        if is_two_words == 1 and token_id + 1 < len(tokens):
                            lang_main[scene_id][object_id][ann_id]["first_obj"] = token_id + 1
                        else:
                            lang_main[scene_id][object_id][ann_id]["first_obj"] = token_id
            if pd == 1:
                lang_main[scene_id][object_id][ann_id]["len"] = len(tokens)
            # store
            lang[scene_id][object_id][ann_id] = embeddings
            lang_main[scene_id][object_id][ann_id]["main"] = main_embeddings
            if scene_id_pre == scene_id:
                i += 1
            else:
                scene_id_pre = scene_id
                i = 0
        return lang, lang_main, lang_len

    def _tranform_bpemb_des(self):
        from bpemb import BPEmb
        self.bpemb = BPEmb(lang="en", vs=self.args.bpemb_vs, dim=self.args.bpemb_dim)        
        
        lang = {}
        lang_len = {}
        lang_main = {}
        scene_id_pre = ""
        i = 0
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            ann_id = data["ann_id"]
            object_name = " ".join(data["object_name"].split("_"))

            if scene_id not in lang:
                lang[scene_id] = {}
                lang_len[scene_id] = {}
                lang_main[scene_id] = {}

            if object_id not in lang[scene_id]:
                lang[scene_id][object_id] = {}
                lang_len[scene_id][object_id] = {}
                lang_main[scene_id][object_id] = {}

            if ann_id not in lang[scene_id][object_id]:
                lang[scene_id][object_id][ann_id] = {}
                lang_len[scene_id][object_id][ann_id] = {}
                lang_main[scene_id][object_id][ann_id] = {}
                lang_main[scene_id][object_id][ann_id]["main"] = {}
                lang_main[scene_id][object_id][ann_id]["len"] = 0
                lang_main[scene_id][object_id][ann_id]["first_obj"] = -1
                lang_main[scene_id][object_id][ann_id]["unk"] = self.bpemb.vectors[0]

            embeddings = np.zeros((self.CONF.TRAIN.MAX_DES_LEN, self.args.bpemb_dim))
            main_embeddings = np.zeros((self.CONF.TRAIN.MAX_DES_LEN, self.args.bpemb_dim))
            pd = 1
            
            desc = data['description']
            tokens = self.bpemb.encode(desc)
            tokens = [re.sub('▁', '', token) for token in tokens]
            lang_len[scene_id][object_id][ann_id] = len(tokens)
            _token_ids = np.array(self.bpemb.encode_ids(desc))[:self.CONF.TRAIN.MAX_DES_LEN]
            embeddings[:len(_token_ids),:] = self.bpemb.vectors[_token_ids]
            main_embeddings[:len(_token_ids),:] = self.bpemb.vectors[_token_ids]            

            main_object_cat = self.raw2label[object_name] if object_name in self.raw2label else self.DC.type2class['others']
            for token_index in range(self.CONF.TRAIN.MAX_DES_LEN):
                if token_index < len(tokens):
                    token = tokens[token_index]
                    if pd == 1:
                        if token == ".":
                            pd = 0
                            lang_main[scene_id][object_id][ann_id]["len"] = token_index + 1
                    object_cat = self.raw2label[token] if token in self.raw2label else -1
                    is_two_words = 0
                    if token_index + 1 < len(tokens):
                        token_new = token + " " + tokens[token_index+1]
                        object_cat_new = self.raw2label[token_new] if token_new in self.raw2label else -1
                        if object_cat_new != -1:
                            object_cat = object_cat_new
                            is_two_words = 1
                    if lang_main[scene_id][object_id][ann_id]["first_obj"] == -1 and object_cat == main_object_cat:
                        if is_two_words == 1 and token_index + 1 < len(tokens):
                            lang_main[scene_id][object_id][ann_id]["first_obj"] = token_index + 1
                        else:
                            lang_main[scene_id][object_id][ann_id]["first_obj"] = token_index

            if pd == 1:
                lang_main[scene_id][object_id][ann_id]["len"] = len(tokens)

            # store
            lang[scene_id][object_id][ann_id] = embeddings
            lang_main[scene_id][object_id][ann_id]["main"] = main_embeddings
            if scene_id_pre == scene_id:
                i += 1
            else:
                scene_id_pre = scene_id
                i = 0
            
        return lang, lang_main, lang_len
    
    
    def _tranform_clip_des(self):
        from clip.simple_tokenizer import SimpleTokenizer
        _tokenizer = SimpleTokenizer()   
        context_length = 77
        
        lang = {}
        lang_len = {}
        lang_main = {}
        scene_id_pre = ""
        i = 0
        
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            ann_id = data["ann_id"]
            object_name = data["object_name"].replace('_', ' ')

            if scene_id not in lang:
                lang[scene_id] = {}
                lang_len[scene_id] = {}
                lang_main[scene_id] = {}

            if object_id not in lang[scene_id]:
                lang[scene_id][object_id] = {}
                lang_len[scene_id][object_id] = {}
                lang_main[scene_id][object_id] = {}

            if ann_id not in lang[scene_id][object_id]:
                lang[scene_id][object_id][ann_id] = {}
                lang_len[scene_id][object_id][ann_id] = {}
                lang_main[scene_id][object_id][ann_id] = {}
                lang_main[scene_id][object_id][ann_id]["main"] = {}
                lang_main[scene_id][object_id][ann_id]["len"] = 0
                lang_main[scene_id][object_id][ann_id]["first_obj"] = -1
                lang_main[scene_id][object_id][ann_id]["unk"] = self.text_feature_objcap["unk"]

            desc = data['description']
            # tokenize description 
            def id2token(token_id):
                return _tokenizer.decoder[token_id]
            assert self.CONF.TRAIN.MAX_DES_LEN >= context_length
            # _token_ids: num_tokens (54)
            _token_ids = _tokenizer.encode(desc)[:context_length-2]
            # tokens: num_tokens (54)
            tokens = [id2token(_token_id) for _token_id in _token_ids]
            # add dummy tokens
            _token_ids = [49406] + _token_ids + [49407]
            tokens = ['<bos>'] + tokens + ['<eos>']
            
            lang_len[scene_id][object_id][ann_id] = len(tokens) # (+ bos, eos)
            embeddings = np.zeros((self.CONF.TRAIN.MAX_DES_LEN, self.clip_size))
            main_embeddings = np.zeros((self.CONF.TRAIN.MAX_DES_LEN, self.clip_size))
            pd = 1
            
            # text_feat: num_tokens (54, 512)
            text_feat = self.text_feature_objcap[scene_id][int(object_id)][int(ann_id)]
            assert text_feat.shape[0] == len(tokens)

            main_object_cat = self.raw2label[object_name] if object_name in self.raw2label else self.DC.type2class['others']
            for token_idx in range(self.CONF.TRAIN.MAX_DES_LEN):
                if token_idx < len(tokens):
                    token = tokens[token_idx]
                    if token in _tokenizer.encoder: # dictionary
                        embeddings[token_idx] = text_feat[token_idx]
                    else:
                        embeddings[token_idx] = self.text_feature_objcap["pad"]
                    if pd == 1:
                        if token in _tokenizer.encoder:
                            main_embeddings[token_idx] = text_feat[token_idx]
                        else:
                            main_embeddings[token_idx] = self.text_feature_objcap["unk"]
                        if token == ".</w>":
                            pd = 0
                            lang_main[scene_id][object_id][ann_id]["len"] = token_idx + 1
                    object_cat = self.raw2label[token.replace('</w>', '')] if token.replace('</w>', '') in self.raw2label else -1
                    is_two_words = 0
                    if token_idx + 1 < len(tokens):
                        token_new = token.replace('</w>', '') + " " + tokens[token_idx+1].replace('</w>', '')
                        object_cat_new = self.raw2label[token_new] if token_new in self.raw2label else -1
                        if object_cat_new != -1:
                            object_cat = object_cat_new
                            is_two_words = 1
                    if lang_main[scene_id][object_id][ann_id]["first_obj"] == -1 and object_cat == main_object_cat:
                        if is_two_words == 1 and token_idx + 1 < len(tokens):
                            lang_main[scene_id][object_id][ann_id]["first_obj"] = token_idx + 1
                        else:
                            lang_main[scene_id][object_id][ann_id]["first_obj"] = token_idx

            if pd == 1:
                lang_main[scene_id][object_id][ann_id]["len"] = len(tokens)

            # store
            lang[scene_id][object_id][ann_id] = embeddings
            lang_main[scene_id][object_id][ann_id]["main"] = main_embeddings
            if scene_id_pre == scene_id:
                i += 1
            else:
                scene_id_pre = scene_id
                i = 0

        return lang, lang_main, lang_len

    def _tranform_bert_des(self):
        from transformers import AutoTokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        _tokenizer = AutoTokenizer.from_pretrained(self.args.bert_model)        
        
        lang = {}
        lang_len = {}
        
        def pad_tokens(tokens):
            N = self.CONF.TRAIN.MAX_DES_LEN - 2 
            if tokens.ndim == 2:
                tokens = tokens[0]
            padded_tokens = np.zeros(self.CONF.TRAIN.MAX_DES_LEN)
            tokens = np.append(tokens[:-1][:N+1], tokens[-1:])
            padded_tokens[:len(tokens)] = tokens
            return padded_tokens
        
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            ann_id = data["ann_id"]

            if scene_id not in lang:
                lang[scene_id] = {}
                lang_len[scene_id] = {}

            if object_id not in lang[scene_id]:
                lang[scene_id][object_id] = {}
                lang_len[scene_id][object_id] = {}

            if ann_id not in lang[scene_id][object_id]:
                lang[scene_id][object_id][ann_id] = {}
                lang_len[scene_id][object_id][ann_id] = {}

            desc = data['description']
            tokens = _tokenizer(desc, return_tensors='np')
            lang_len[scene_id][object_id][ann_id] = tokens['input_ids'].shape[1] 
            
            # for BERT
            if 'token_type_ids' in tokens:
                padded_input_ids = pad_tokens(tokens['input_ids'])
                padded_token_type_ids = pad_tokens(tokens['token_type_ids'])
                padded_attention_mask = pad_tokens(tokens['attention_mask'])
                # store
                lang[scene_id][object_id][ann_id] = {
                    'input_ids': padded_input_ids, 
                    'token_type_ids': padded_token_type_ids,
                    'attention_mask': padded_attention_mask,
                }
            else: # for DistillBERT
                padded_input_ids = pad_tokens(tokens['input_ids'])
                padded_attention_mask = pad_tokens(tokens['attention_mask'])
                lang[scene_id][object_id][ann_id] = {
                    'input_ids': padded_input_ids, 
                    'attention_mask': padded_attention_mask,
                }
        return lang, lang_len
    
            
    def _load_data(self):
        print("loading data...")
        # add scannet data
        self.scene_list = sorted(list(set([data["scene_id"] for data in self.scanrefer])))
        self.scene_data = {}
    
        scene_data = joblib.Parallel(n_jobs=3, prefer="threads")(
            joblib.delayed(_load_data_by_scene)(scene_id, self.CONF.PATH.SCAN_DATA) for scene_id in self.scene_list)
        
        # load pose data
        pose_file = os.path.join(self.CONF.PATH.META, 'camera_pose.pkl')
        with open(pose_file, 'rb') as f:
            camera_pose_dic = pickle.load(f)    
        
        for (scene_id, scene_info) in scene_data:
            self.scene_data[scene_id] = scene_info
            self.scene_data[scene_id]['camera_poses'] = camera_pose_dic[scene_id]
        
        # prepare class mapping
        self.raw2label = self._get_raw2label()
        self.unique_multiple_lookup = self._get_unique_multiple_lookup()

        # load language features
        if self.args.tokenizer == 'glove':
            self.lang, self.lang_main, self.lang_len = self._tranform_glove_des()
        elif self.args.tokenizer == 'bpemb':
            self.lang, self.lang_main, self.lang_len = self._tranform_bpemb_des()
        elif self.args.tokenizer == 'clip':
            assert self.args.use_token_clip
            self.lang, self.lang_main, self.lang_len = self._tranform_clip_des()            
        elif self.args.tokenizer == 'bert':
            self.lang, self.lang_len = self._tranform_bert_des()                
        else:
            raise NotImplementedError

    def _translate(self, point_set, bbox):
        # unpack
        coords = point_set[:, :3]

        # translation factors
        x_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        y_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        z_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        factor = [x_factor, y_factor, z_factor]
        
        # dump
        coords += factor
        point_set[:, :3] = coords
        bbox[:, :3] += factor

        return point_set, bbox
