import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pickle
import sys
import os
import argparse
import glob
import json
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

# Load external constants

sys.path.append("../scannet")
from scannet200_constants import CLASS_LABELS_20, CLASS_LABELS_200, VALID_CLASS_IDS_200
from scannet200_splits import HEAD_CATS_SCANNET_200, COMMON_CATS_SCANNET_200, TAIL_CATS_SCANNET_200
from utils import read_plymesh, compute_normal, point_indices_from_group, save_plymesh


CLOUD_FILE_PFIX = 'mesh.refined.v2.color'
AGGREGATIONS_FILE_PFIX = 'semseg.v2.json'
SEGMENTS_FILE_PFIX = 'mesh.refined.0.010000.segs.v2.json'
CLASS_IDs = VALID_CLASS_IDS_200

MAX_NUM_POINT = 50000

def read_transform_matrix(Scan3RJson_PATH):
    rescan2ref = {}
    with open(Scan3RJson_PATH , "r") as read_file:
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

def handle_process(scene_path, output_path, save_mesh, labels_pd, train_scenes, val_scenes, test_scenes):
    scene_id = scene_path.split('/')[-1]
    mesh_path = os.path.join(scene_path, f'{CLOUD_FILE_PFIX}.ply')    
    aggregations_file = os.path.join(scene_path, f'{AGGREGATIONS_FILE_PFIX}')
    segments_file = os.path.join(scene_path, f'{SEGMENTS_FILE_PFIX}')
    # Rotating the mesh to axis aligned
    # axis_align_matrix = rescan2ref.get(scene_id, np.identity(4))
    rot_matrix = rescan2ref.get(scene_id, np.identity(4))
    
    ref_scene_id = reference_dic[scene_id]
    ref_rot_matrix = reference_axis_align_matrix_dic[ref_scene_id]
    
    if scene_id in train_scenes:
        split_name = 'train'
    elif scene_id in val_scenes:
        split_name = 'val'
    elif scene_id in test_scenes:
        split_name = 'test'
    else:
        print('*', scene_id, 'does not exist in [train, val, test] that have seg files')
        return 

    print('Processing: ', scene_id, 'in', split_name)
    # pointcloud: num_points, 7
    pointcloud, faces_array = read_plymesh(mesh_path)
    points = pointcloud[:, :3]
    colors = pointcloud[:, 3:6]
    alphas = pointcloud[:, -1]
    # compute normal
    normals = compute_normal(points, faces_array)

    # Rotate PC to axis aligned
    r_points = pointcloud[:, :3].transpose()
    r_points = np.append(r_points, np.ones((1, r_points.shape[1])), axis=0)
    # reference align
    r_points = np.dot(rot_matrix, r_points)
    # reference axis align
    r_points = np.dot(ref_rot_matrix, r_points)
    aligned_pointcloud = np.append(r_points.transpose()[:, :3], pointcloud[:, 3:], axis=1)

    # Generate new labels
    labelled_pc = np.zeros((pointcloud.shape[0], 1)) # 0: unannotated
    instance_ids = np.zeros((pointcloud.shape[0], 1)) # 0: unannotated
        
    if os.path.isfile(aggregations_file):
        # Load segments file
        with open(segments_file) as f:
            segments = json.load(f)
            seg_indices = np.array(segments['segIndices'])        
        # Load Aggregations file
        with open(aggregations_file) as f:
            aggregation = json.load(f)
            seg_groups = np.array(aggregation['segGroups'])

        #num_verts = len(data['segIndices'])
        num_instances = len(seg_groups)        
        instance_bboxes = np.zeros((num_instances, 8)) # also include object id
        aligned_instance_bboxes = np.zeros((num_instances, 8)) # also include object id
            
        for obj_idx, group in enumerate(seg_groups):
            segment_points, aligned_segment_points, p_inds, label_id = point_indices_from_group(pointcloud, aligned_pointcloud, seg_indices, group, labels_pd, CLASS_IDs)
            labelled_pc[p_inds] = label_id
            
            if len(segment_points) == 0: continue
            # Compute axis aligned box
            # An axis aligned bounding box is parameterized by
            # (cx,cy,cz) and (dx,dy,dz) and label id
            # where (cx,cy,cz) is the center point of the box,
            # dx is the x-axis length of the box.
            xmin = np.min(segment_points[:,0])
            ymin = np.min(segment_points[:,1])
            zmin = np.min(segment_points[:,2])
            xmax = np.max(segment_points[:,0])
            ymax = np.max(segment_points[:,1])
            zmax = np.max(segment_points[:,2])
            bbox = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2, xmax-xmin, ymax-ymin, zmax-zmin, label_id, group['id']]) # also include object id
            # NOTE: this assumes obj_id is in 1,2,3,.,,,.NUM_INSTANCES
            # instance_bboxes[group['id'],:] = bbox 
            # Note that for 3RScan, object_id does not match the number of segments.
            instance_bboxes[obj_idx,:] = bbox 
            
            if len(aligned_segment_points) == 0: continue
            #instance_ids[p_inds] = group['id'] # should equal to group['objectId'] 
            instance_ids[p_inds] = group['id'] + 1 
            xmin = np.min(aligned_segment_points[:,0])
            ymin = np.min(aligned_segment_points[:,1])
            zmin = np.min(aligned_segment_points[:,2])
            xmax = np.max(aligned_segment_points[:,0])
            ymax = np.max(aligned_segment_points[:,1])
            zmax = np.max(aligned_segment_points[:,2])
            bbox = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2, xmax-xmin, ymax-ymin, zmax-zmin, label_id, group['id']]) # also include object id
            # NOTE: this assumes group['id'] is in 1, 2, ...NUM_INSTANCES
            #aligned_instance_bboxes[group['id'],:] = bbox 
            aligned_instance_bboxes[obj_idx,:] = bbox 
    else:
        # use zero as placeholders for the test scene
        #print("use placeholders")
        instance_bboxes = np.zeros((1, 8)) # also include object id
        aligned_instance_bboxes = np.zeros((1, 8)) # also include object id

    labelled_pc = labelled_pc.astype(int)
    instance_ids = instance_ids.astype(int)        
    # Concatenate with original cloud
    # processed_vertices: num_points, 11 (xyz, color, normal, label, instance_id)
    processed_vertices = np.hstack((pointcloud[:, :6], normals, labelled_pc, instance_ids))        
    aligned_processed_vertices = np.hstack((aligned_pointcloud[:, :6], normals, labelled_pc, instance_ids))        
    
    if (np.any(np.isnan(processed_vertices)) or not np.all(np.isfinite(processed_vertices))):
        raise ValueError('nan')
    # Save processed mesh
    if save_mesh:
        output_file = os.path.join(output_path, f'{scene_id}.ply')
        output_aligned_file = os.path.join(output_path, f'{scene_id}_aligned.ply')
        save_plymesh(processed_vertices, faces_array, output_file, with_label=True, verbose=False)
        save_plymesh(aligned_processed_vertices, faces_array, output_aligned_file, with_label=True, verbose=False)        
    # Uncomment the following lines if saving the output in voxelized point cloud
    # quantized_points, quantized_scene_colors, quantized_labels, quantized_instances = voxelize_pointcloud(points, colors, labelled_pc, instance_ids, faces_array)
    # quantized_pc = np.hstack((quantized_points, quantized_scene_colors, quantized_labels, quantized_instances))
    # save_plymesh(quantized_pc, faces=None, filename=output_file, with_label=True, verbose=False)    
    
    #'''
    N = processed_vertices.shape[0]    
    if N > MAX_NUM_POINT:
        choices = np.random.choice(N, MAX_NUM_POINT, replace=False)
        processed_vertices = processed_vertices[choices, :]
        aligned_processed_vertices = aligned_processed_vertices[choices, :]
        labelled_pc = labelled_pc[choices]
        instance_ids = instance_ids[choices]
    #'''

    #print("Shape of points: {}".format(processed_vertices.shape))    
    
    output_prefix = os.path.join(output_path, f'{scene_id}')
    np.save(output_prefix+'_vert.npy', processed_vertices)
    np.save(output_prefix+'_aligned_vert.npy', aligned_processed_vertices)
    np.save(output_prefix+'_sem_label.npy', labelled_pc)
    np.save(output_prefix+'_ins_label.npy', instance_ids)
    np.save(output_prefix+'_bbox.npy', instance_bboxes)
    # np.save(output_file+'_aligned_bbox.npy', instance_bboxes) 
    np.save(output_prefix+'_aligned_bbox.npy', aligned_instance_bboxes)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='scans', help='Path to the 3RScan dataset containing scene folders')
    parser.add_argument('--output_root', default='rio200_data', help='Output path where processed data will be located')
    #parser.add_argument('--label_map_file', default='meta_data/3RScan.v2_Semantic-Classes-Mapping.csv', help='path to scannetv2-labels.combined.tsv')
    parser.add_argument('--label_map_file', default='../scannet/meta_data/scannetv2-labels.combined.tsv', help='path to scannetv2-labels.combined.tsv')
    parser.add_argument('--num_workers', default=12, type=int, help='The number of parallel workers')
    parser.add_argument('--splits_path', default='meta_data/split', help='Where the txt files with the train/val splits live')
    parser.add_argument('--save_mesh', action='store_true', help='save mesh file')
    config = parser.parse_args()

    # Load label map
    #labels_pd = pd.read_csv(config.label_map_file, sep=',', header=1)
    labels_pd = pd.read_csv(config.label_map_file, sep='\t', header=0)

    # Load train/val splits
    with open(config.splits_path + '/3rscan_train.txt') as train_file:
        train_scenes = train_file.read().splitlines()
    with open(config.splits_path + '/3rscan_val.txt') as val_file:
        val_scenes = val_file.read().splitlines()
    with open(config.splits_path + '/3rscan_test.txt') as test_file:
        test_scenes = test_file.read().splitlines()
        
    # Create output directories
    # train_output_dir = os.path.join(config.output_root, 'train')
    # if not os.path.exists(train_output_dir):
    #     os.makedirs(train_output_dir)
    # val_output_dir = os.path.join(config.output_root, 'val')
    # if not os.path.exists(val_output_dir):
    #     os.makedirs(val_output_dir)
    # test_output_dir = os.path.join(config.output_root, 'test')
    # if not os.path.exists(test_output_dir):
    #     os.makedirs(test_output_dir)
    
    META_FILE = '/home/dbi-data7/miyanishi/Data/3RScan/3RScan.json'
    rescan2ref = read_transform_matrix(META_FILE)
    reference_dic = get_reference_dic(META_FILE)
    
    with open('./meta_data/reference_axis_align_matrix.pkl', 'rb') as f:
        reference_axis_align_matrix_dic = pickle.load(f)

    os.makedirs(config.output_root, exist_ok=True)

    # Load scene paths
    scene_paths = sorted(glob.glob(config.dataset_root + '/*'))
    
    # Preprocess data.
    pool = ProcessPoolExecutor(max_workers=config.num_workers)
    print('Processing scenes...')
    _ = list(pool.map(handle_process, scene_paths, repeat(config.output_root), repeat(config.save_mesh), repeat(labels_pd), repeat(train_scenes), repeat(val_scenes), repeat(test_scenes)))
