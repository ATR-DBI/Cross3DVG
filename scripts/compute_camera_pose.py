import re,sys,os
import glob
import argparse
import numpy as np
import pandas as pd
import json
import pickle

from collections import OrderedDict

from data.scannet.model_util_scannet import ScannetDatasetConfig

def get_rot_matrix(axis_file):
    rot_matrix = np.identity(4)
    lines = open(axis_file).readlines()
    for line in lines:
        if 'axisAlignment' in line:
            rot_matrix = np.array([[float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]]).reshape(4, 4)
            break
    return rot_matrix


def read_camera_pose(filename):
    file = open(filename, 'r')
    data = file.read()
    lines = data.split("\n")
    pose = np.array([[float(v.strip()) for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"])
    return pose


# camera_pose by scene
def get_camera_poses(pose_files, rot_matrix=None, ref_rot_matrix=None):
    camera_pose_dic = {}
    if args.dataset == 'scanrefer':
        for pose_file in pose_files:
            # pose_file: 0.txt, 1280.txt
            pose_id = int(os.path.basename(pose_file).strip('.txt'))
            camera_pose = read_camera_pose(pose_file)
            if rot_matrix is not None:
                camera_pose = np.dot(rot_matrix, camera_pose)
            camera_pose_dic[pose_id] = camera_pose
    elif 'riorefer' in args.dataset:
        for pose_file in pose_files:
            # pose_file: frame-000044.pose.txt
            pose_id = os.path.basename(pose_file).replace('.txt', '').replace('.pose', '').replace('.aligned', '')
            camera_pose = read_camera_pose(pose_file)
            camera_pose = np.dot(rot_matrix, camera_pose)                
            camera_pose = np.dot(ref_rot_matrix, camera_pose)
            camera_pose_dic[pose_id] = camera_pose
    else:
        raise NotImplementedError
    # sort with pose_id
    camera_pose_dic = OrderedDict(sorted(camera_pose_dic.items()))        
    pose_ids = np.stack(list(camera_pose_dic.keys()))  
    return camera_pose_dic


def read_transform_matrix(meta_file):
    rescan2ref = {}
    with open(meta_file, "r") as read_file:
        data = json.load(read_file)
        for scene in data:
            for scans in scene["scans"]:
                if "transform" in scans:
                    rescan2ref[scans["reference"]] = np.array(scans["transform"]).reshape(4,4).T
    return rescan2ref    


def get_reference_dic(Scan3RJson_PATH):
    meta_data = json.load(open(Scan3RJson_PATH))
    reference_dic = {}
    for record in meta_data:
        reference = record['reference'] 
        reference_dic[reference] = reference
        if 'scans' not in record:
            continue
        for scan in record['scans']:
            reference_dic[scan['reference']] = reference
    return reference_dic


def get_camera_pose_dic(args, CONF, scene_list):
    camera_pose_dic = {}
    if 'riorefer' in args.dataset:
        meta_file = os.path.join(CONF.PATH.META, "3RScan.json")
        rescan2ref = read_transform_matrix(meta_file)    
        reference_dic = get_reference_dic(meta_file)
        
        with open(os.path.join(CONF.PATH.META, 'reference_axis_align_matrix.pkl'), 'rb') as f:
            reference_axis_align_matrix_dic = pickle.load(f)        
    
    for j, scene_id in enumerate(scene_list):
        # axis_align_file is needed if use _aligned_bbox
        if args.dataset == 'scanrefer':
            axis_align_file = os.path.join(args.scanet_scans, scene_id, scene_id + '.txt')
            rot_matrix = get_rot_matrix(axis_align_file)                
            # camera pose            
            pose_dir = os.path.join(CONF.PATH.FRAME_SQUARE, scene_id, "pose")
            pose_files = [os.path.join(pose_dir, pose_fname) for pose_fname in os.listdir(pose_dir)]
            camera_pose_dic[scene_id] = get_camera_poses(pose_files, rot_matrix)
        elif 'riorefer' in args.dataset:
            rot_matrix = rescan2ref.get(scene_id, np.identity(4))            
            ref_scene_id = reference_dic[scene_id]
            ref_rot_matrix = reference_axis_align_matrix_dic[ref_scene_id]
            # camera pose            
            pose_dir = os.path.join(CONF.PATH.SCANS, scene_id, "sequence")
            # num sampling frames is 5
            pose_files = sorted(glob.glob(os.path.join(pose_dir, '*[!align].pose.txt')))[::args.pose_step]
            camera_pose_dic[scene_id] = get_camera_poses(pose_files, rot_matrix, ref_rot_matrix)
        else:
            raise NotImplementedError                    
        print(j, '/', len(scene_list), scene_id)
        
    if args.dataset == 'scanrefer':    
        output_file = 'data/scannet/meta_data/camera_pose.pkl'
    elif args.dataset == 'riorefer':
        output_file = 'data/rio/meta_data/camera_pose.pkl'
        
    with open(output_file, 'wb') as f:
        pickle.dump(camera_pose_dic, f)
        
        
def run(args):   
    scene_list = []     
    if args.dataset == 'scanrefer':
        from lib.scanrefer_config import CONF
        scanrefer_dir = 'data/scanrefer'
        for split in ['train', 'val', 'test']:
            scanrefer_file = os.path.join(scanrefer_dir, 'ScanRefer_filtered_'+split+'.json')
            scene_list += list(pd.read_json(scanrefer_file).scene_id)
    elif 'riorefer' in args.dataset:
        from lib.riorefer_config import CONF
        riorefer_dir = 'data/rio/meta_data/split'
        for split in ['train', 'val', 'test']:
            scanrefer_file = os.path.join(riorefer_dir, '3rscan_'+split+'.txt')
            scene_list += list([line.strip() for line in open(scanrefer_file).readlines()])
    else:
        raise NotImplementedError
    scene_list  = sorted(set(scene_list))        
    get_camera_pose_dic(args, CONF, scene_list)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset 
    parser.add_argument("--dataset", type=str, help="refer dataset", default="scanrefer") # scanrefer or riorefer
    parser.add_argument("--scanet_scans", type=str, help="refer dataset", default="data/scannet/scans")     
    parser.add_argument("--pose_step", type=int, help="", default=5) # head, head_common, head_common_tail
    args = parser.parse_args()
    
    run(args)
