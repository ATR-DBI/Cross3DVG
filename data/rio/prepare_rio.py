import os
import sys
import datetime
import numpy as np
import pdb

import math
import os, sys, argparse
import inspect
import json
import pdb
import csv
import numpy as np
import pandas as pd


try:
    from plyfile import PlyData, PlyElement
except:
    print("Please install the module 'plyfile' for PLY i/o, e.g.")
    print("pip install plyfile")
    sys.exit(-1)
    
    
def parse_args():
    parser = argparse.ArgumentParser('Data Preparision')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--scannet_path', type=str, default='scans/')
    #parser.add_argument('--scannet_path', type=str, default='data/scannet/scans/')
    #parser.add_argument('--pointgroupinst_path', type=str, default='PointGroupInst/')
    #parser.add_argument('--output_path', type=str, default='pointgroup_data')
    parser.add_argument('--output_path', type=str, default='rio_data')
    return parser.parse_args()
    

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
    arr[:,0] /= (lens + 1e-8)
    arr[:,1] /= (lens + 1e-8)
    arr[:,2] /= (lens + 1e-8)                
    return arr

def compute_normal(vertices, faces):
    #Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    normals = np.zeros( vertices.shape, dtype=vertices.dtype )
    #Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    #Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle             
    n = np.cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] )
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices, 
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle, 
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    normals[ faces[:,0] ] += n
    normals[ faces[:,1] ] += n
    normals[ faces[:,2] ] += n
    normalize_v3(normals)
    
    return normals

def represents_int(s):
    ''' if string s represents an int. '''
    try: 
        int(s)
        return True
    except ValueError:
        return False

def read_mesh_vertices(filename):
    """ read XYZ for each vertex.
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
    return vertices

def read_mesh_vertices_rgb(filename):
    """ read XYZ RGB for each vertex.
    Note: RGB values are in 0-255
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
        vertices[:,3] = plydata['vertex'].data['red']
        vertices[:,4] = plydata['vertex'].data['green']
        vertices[:,5] = plydata['vertex'].data['blue']
    return vertices

def read_mesh_vertices_rgb_normal(filename):
    """ read XYZ RGB normals point cloud from filename PLY file """
    assert(os.path.isfile(filename))
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 9], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
        vertices[:,3] = plydata['vertex'].data['red']
        vertices[:,4] = plydata['vertex'].data['green']
        vertices[:,5] = plydata['vertex'].data['blue']

        #print('*', plydata["vertex"].data[0])
        #exit()
        # compute normals
        #xyz = np.array([[x, y, z] for x, y, z, _, _, _, _ in plydata["vertex"].data])
        # 3rscanはx,y,z,r,g,b,nx,ny,nz
        xyz = np.array([[x, y, z] for x, y, z, _, _, _, _, _, _ in plydata["vertex"].data])
        face = np.array([f[0] for f in plydata["face"].data])
        nxnynz = compute_normal(xyz, face)
        vertices[:,6:] = nxnynz
    return vertices

# def read_label_mapping(filename, label_from='raw_category', label_to='nyu40id'):
#     assert os.path.isfile(filename)
#     mapping = dict()
#     with open(filename) as csvfile:
#         reader = csv.DictReader(csvfile, delimiter='\t')
#         for row in reader:
#             mapping[row[label_from]] = int(row[label_to])
#     if represents_int(list(mapping.keys())[0]):
#         mapping = {int(k):v for k,v in mapping.items()}
#     return mapping

def read_3rscan_label_mapping(label_map_file, label_from='Label', label_to='nyuid', skip=0):
    label_df = pd.read_csv(label_map_file, skiprows=1)
    if label_to == 'labelid':
        label_to = 'Global ID'    
    elif label_to == 'nyu40id':
        label_to = 'Unnamed: 2'
    elif label_to == 'eigenid':
        label_to = 'Unnamed: 4'
    elif label_to == 'rio27':
        label_to = 'Unnamed: 6'    
    elif label_to == 'rio7':
        label_to = 'Unnamed: 8'        
    else:
        raise NotImplemented
    mapping = {row[label_from]:row[label_to] for _, row in label_df.iterrows()}
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k):v for k,v in mapping.items()}
    return mapping


def read_aggregation(filename):
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        for i in range(num_objects):
            object_id = data['segGroups'][i]['objectId'] + 1 # instance ids should be 1-indexed
            label = data['segGroups'][i]['label']
            segs = data['segGroups'][i]['segments']
            object_id_to_segs[object_id] = segs
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs


def read_segmentation(filename):
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data['segIndices'])
        for i in range(num_verts):
            seg_id = data['segIndices'][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts

def read_transform_matrix(Scan3RJson_PATH):
    rescan2ref = {}
    with open(Scan3RJson_PATH , "r") as read_file:
        data = json.load(read_file)
        for scene in data:
            for scans in scene["scans"]:
                if "transform" in scans:
                    #rescan2ref[scans["reference"]] = np.matrix(scans["transform"]).reshape(4,4)
                    #rescan2ref[scans["reference"]] = np.array(scans["transform"]).reshape(4,4)
                    rescan2ref[scans["reference"]] = np.array(scans["transform"]).reshape(4,4).T
    return rescan2ref

#def export(mesh_file, agg_file, seg_file, meta_file, label_map_file, output_file=None, pointgroup_file=None):
def export(scan_name, mesh_file, agg_file, seg_file, label_map_file, output_file=None):
    """ points are XYZ RGB (RGB in 0-255),
    semantic label as nyu40 ids,
    instance label as 1-#instance,
    box as (cx,cy,cz,dx,dy,dz,semantic_label)
    """
    #scene = meta_file.split('/')[-1].split('.')[0]

    # if split == 'train':
    #     try:
    #         temp_dir = pointgroup_file + '/train/'
    #         inst_list = pd.read_table(temp_dir + scene + '.txt', header=None)
    #     except:
    #         temp_dir = pointgroup_file + '/val/'
    #         inst_list = pd.read_table(temp_dir + scene + '.txt', header=None)
    # else:
    #     temp_dir = pointgroup_file + '/test/'
    #     inst_list = pd.read_table(temp_dir + scene + '.txt', header=None)

    #label_map = scannet_utils.read_label_mapping(label_map_file, label_from='raw_category', label_to='nyu40id')
    #mesh_vertices = scannet_utils.read_mesh_vertices_rgb_normal(mesh_file)
    #label_map = read_label_mapping(label_map_file, label_from='raw_category', label_to='nyu40id')    
    label_map = read_3rscan_label_mapping(label_map_file, label_from='Label', label_to='nyu40id', skip=1)
    #print(scan_name) # 095821f7-e2c2-2de1-9568-b9ce59920e29
    #print(mesh_file) # scans/095821f7-e2c2-2de1-9568-b9ce59920e29/mesh.refined.v2.color.ply
    #exit()
    mesh_vertices = read_mesh_vertices_rgb_normal(mesh_file)    

    # Load scene axis alignment matrix
    #lines = open(meta_file).readlines()
    #axis_align_matrix = None
    # for line in lines:
    #     if 'axisAlignment' in line:
    #         axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
    axis_align_matrix = rescan2ref.get(scan_name, None)
    
    if axis_align_matrix is not None:
        axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
        # axis_align_matrix
        # [[ 0.94303304 -0.33265644  0.00531485  0.        ]
        # [ 0.33258364  0.94300646  0.0112634   0.        ]
        # [-0.00875879 -0.00885413  0.99992234  0.        ]
        # [-0.42789239  0.47738436 -0.10655134  1.        ]]
        pts = np.ones((mesh_vertices.shape[0], 4))
        pts[:, 0:3] = mesh_vertices[:, 0:3]
        pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4
        aligned_vertices = np.copy(mesh_vertices)
        aligned_vertices[:, 0:3] = pts[:, 0:3]
    else:
        print("No axis alignment matrix found")
        aligned_vertices = mesh_vertices

    # Load semantic and instance labels
    if os.path.isfile(agg_file):
        object_id_to_segs, label_to_segs = read_aggregation(agg_file)
        seg_to_verts, num_verts = read_segmentation(seg_file)

        label_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated
        object_id_to_label_id = {}
        for label, segs in label_to_segs.items():
            #print('*', label, label_map[label])
            #exit()
            label_id = label_map[label]
            for seg in segs:
                verts = seg_to_verts[seg]
                label_ids[verts] = label_id
        instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated
        num_instances = len(np.unique(list(object_id_to_segs.keys())))
        for object_id, segs in object_id_to_segs.items():
            for seg in segs:
                verts = seg_to_verts[seg]
                instance_ids[verts] = object_id
                if object_id not in object_id_to_label_id:
                    object_id_to_label_id[object_id] = label_ids[verts][0]

        instance_bboxes = np.zeros((num_instances, 8))  # also include object id
        aligned_instance_bboxes = np.zeros((num_instances, 8))  # also include object id

        for obj_idx, obj_id in enumerate(object_id_to_segs):
            label_id = object_id_to_label_id[obj_id]

            # bboxes in the original meshes
            obj_pc = mesh_vertices[instance_ids == obj_id, 0:3]
            if len(obj_pc) == 0: continue
            # Compute axis aligned box
            # An axis aligned bounding box is parameterized by
            # (cx,cy,cz) and (dx,dy,dz) and label id
            # where (cx,cy,cz) is the center point of the box,
            # dx is the x-axis length of the box.
            xmin = np.min(obj_pc[:, 0])
            ymin = np.min(obj_pc[:, 1])
            zmin = np.min(obj_pc[:, 2])
            xmax = np.max(obj_pc[:, 0])
            ymax = np.max(obj_pc[:, 1])
            zmax = np.max(obj_pc[:, 2])
            bbox = np.array(
                [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2, xmax - xmin, ymax - ymin, zmax - zmin,
                 label_id, obj_id - 1])  # also include object id (read_aggregation内でobject_id+1してるので、戻すために-1しておく)
            #print('scan_name', scan_name, 'obj_idx', obj_idx, 'obj_id', obj_id - 1)
            # scan_name 0988ea78-eb32-2e61-80ee-e4a44170bce9 obj_idx 0 obj_id 8
            # scan_name 0988ea78-eb32-2e61-80ee-e4a44170bce9 obj_idx 1 obj_id 9
            # scan_name 0988ea78-eb32-2e61-80ee-e4a44170bce9 obj_idx 2 obj_id 10
            # scan_name 0988ea78-eb32-2e61-80ee-e4a44170bce9 obj_idx 3 obj_id 11
            # scan_name 0988ea78-eb32-2e61-80ee-e4a44170bce9 obj_idx 4 obj_id 17
            
            # NOTE: this assumes obj_id is in 1,2,3,.,,,.NUM_INSTANCES
            #instance_bboxes[obj_id - 1, :] = bbox
            # 3RScanはobject_idがセグメントの個数と一致しないので注意
            instance_bboxes[obj_idx,:] = bbox 

            # bboxes in the aligned meshes
            obj_pc = aligned_vertices[instance_ids == obj_id, 0:3]
            if len(obj_pc) == 0: continue
            # Compute axis aligned box
            # An axis aligned bounding box is parameterized by
            # (cx,cy,cz) and (dx,dy,dz) and label id
            # where (cx,cy,cz) is the center point of the box,
            # dx is the x-axis length of the box.
            xmin = np.min(obj_pc[:, 0])
            ymin = np.min(obj_pc[:, 1])
            zmin = np.min(obj_pc[:, 2])
            xmax = np.max(obj_pc[:, 0])
            ymax = np.max(obj_pc[:, 1])
            zmax = np.max(obj_pc[:, 2])
            bbox = np.array(
                [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2, xmax - xmin, ymax - ymin, zmax - zmin,
                 label_id, obj_id - 1])  # also include object id
            # NOTE: this assumes obj_id is in 1,2,3,.,,,.NUM_INSTANCES
            #aligned_instance_bboxes[obj_id - 1, :] = bbox
            aligned_instance_bboxes[obj_idx,:] = bbox             
    else:
        # use zero as placeholders for the test scene
        print("use placeholders")
        num_verts = mesh_vertices.shape[0]
        label_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated
        instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated
        instance_bboxes = np.zeros((1, 8))  # also include object id
        aligned_instance_bboxes = np.zeros((1, 8))  # also include object id

    # label_ids_pg = np.zeros(shape=(num_verts), dtype=np.uint32)
    # instance_ids_pg = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated

    # for inst_id, inst_pg in enumerate(inst_list[0]):
    #     txt_path, cls, _ = inst_pg.split(' ')
    #     inst_pred = np.loadtxt(os.path.join(temp_dir, txt_path))
    #     instance_ids_pg[inst_pred != 0] = inst_id + 1
    #     label_ids_pg[inst_pred != 0] = cls

    if output_file is not None:
        np.save(output_file + '_vert.npy', mesh_vertices)
        np.save(output_file + '_aligned_vert.npy', aligned_vertices)
        np.save(output_file + '_sem_label.npy', label_ids)
        np.save(output_file + '_ins_label.npy', instance_ids)
        #np.save(output_file + '_sem_label_pg.npy', label_ids_pg)
        #np.save(output_file + '_ins_label_pg.npy', instance_ids_pg)
        np.save(output_file + '_bbox.npy', instance_bboxes)
        np.save(output_file + '_aligned_bbox.npy', instance_bboxes)

    #return mesh_vertices, aligned_vertices, label_ids, instance_ids, instance_bboxes, aligned_instance_bboxes, label_ids_pg, instance_ids_pg
    return mesh_vertices, aligned_vertices, label_ids, instance_ids, instance_bboxes, aligned_instance_bboxes



def export_one_scan(scan_name, output_filename_prefix):
    # mesh_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '_vh_clean_2.ply')
    # # agg_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '_vh_clean.aggregation.json')
    # agg_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '.aggregation.json') # NOTE must use the aggregation file for the low-res mesh
    # seg_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '_vh_clean_2.0.010000.segs.json')
    mesh_file = os.path.join(SCANNET_DIR, scan_name, 'mesh.refined.v2.color.ply')
    agg_file = os.path.join(SCANNET_DIR, scan_name, 'semseg.v2.json')
    seg_file = os.path.join(SCANNET_DIR, scan_name, 'mesh.refined.0.010000.segs.v2.json')

    meta_file = os.path.join(SCANNET_DIR, scan_name,
                             scan_name + '.txt')  # includes axisAlignment info for the train set scans.

    # mesh_vertices, aligned_vertices, semantic_labels, instance_labels, \
    # instance_bboxes, aligned_instance_bboxes, semantic_labels_pg, instance_labels_pg = \
    #     export(mesh_file, agg_file, seg_file, meta_file, LABEL_MAP_FILE, None, POINTGROUP_DIR)
        
    mesh_vertices, aligned_vertices, semantic_labels, instance_labels, instance_bboxes, aligned_instance_bboxes = \
        export(scan_name, mesh_file, agg_file, seg_file, LABEL_MAP_FILE, None)        

    mask = np.logical_not(np.in1d(semantic_labels, DONOTCARE_CLASS_IDS))
    mesh_vertices = mesh_vertices[mask, :]
    aligned_vertices = aligned_vertices[mask, :]
    semantic_labels = semantic_labels[mask]
    instance_labels = instance_labels[mask]

    if instance_bboxes.shape[0] > 1:
        num_instances = len(np.unique(instance_labels))
        print('Num of instances: ', num_instances)

        # bbox_mask = np.in1d(instance_bboxes[:,-1], OBJ_CLASS_IDS)
        bbox_mask = np.in1d(instance_bboxes[:, -2], OBJ_CLASS_IDS)  # match the mesh2cap
        instance_bboxes = instance_bboxes[bbox_mask, :]
        aligned_instance_bboxes = aligned_instance_bboxes[bbox_mask, :]
        print('Num of care instances: ', instance_bboxes.shape[0])
    else:
        print("No semantic/instance annotation for test scenes")

    N = mesh_vertices.shape[0]
    if N > MAX_NUM_POINT:
        choices = np.random.choice(N, MAX_NUM_POINT, replace=False)
        mesh_vertices = mesh_vertices[choices, :]
        aligned_vertices = aligned_vertices[choices, :]
        semantic_labels = semantic_labels[choices]
        instance_labels = instance_labels[choices]
        #semantic_labels_pg = semantic_labels_pg[choices]
        #instance_labels_pg = instance_labels_pg[choices]

    print("Shape of points: {}".format(mesh_vertices.shape))
    # exit()
    np.save(output_filename_prefix + '_vert.npy', mesh_vertices)
    np.save(output_filename_prefix + '_aligned_vert.npy', aligned_vertices)
    np.save(output_filename_prefix + '_sem_label.npy', semantic_labels)
    np.save(output_filename_prefix + '_ins_label.npy', instance_labels)
    #np.save(output_filename_prefix + '_sem_label_pg.npy', semantic_labels_pg)
    #np.save(output_filename_prefix + '_ins_label_pg.npy', instance_labels_pg)
    np.save(output_filename_prefix + '_bbox.npy', instance_bboxes)
    np.save(output_filename_prefix + '_aligned_bbox.npy', aligned_instance_bboxes)


def batch_export():
    if not os.path.exists(OUTPUT_FOLDER):
        print('Creating new data folder: {}'.format(OUTPUT_FOLDER))
        os.mkdir(OUTPUT_FOLDER)

    for scan_name in SCAN_NAMES:
        print(scan_name)
        output_filename_prefix = os.path.join(OUTPUT_FOLDER, scan_name)
        # if os.path.exists(output_filename_prefix + '_vert.npy'): continue

        print('-' * 20 + 'begin')
        print(datetime.datetime.now())
        print(scan_name)

        export_one_scan(scan_name, output_filename_prefix)

        print('-' * 20 + 'done')


if __name__ == '__main__':
    args = parse_args()
    split = args.split
    SCANNET_DIR = args.scannet_path
    #POINTGROUP_DIR = args.pointgroupinst_path
    OUTPUT_FOLDER = args.output_path

    SCAN_NAMES = sorted([line.rstrip() for line in open('meta_data/split/3rscan_%s.txt' % split)])
    LABEL_MAP_FILE = 'meta_data/3RScan.v2_Semantic-Classes-Mapping.csv'
    DONOTCARE_CLASS_IDS = np.array([])
    OBJ_CLASS_IDS = np.array(
        [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
         33, 34, 35, 36, 37, 38, 39, 40])  # exclude wall (1), floor (2), ceiling (22)
    MAX_NUM_POINT = 50000
    META_FILE = '/home/dbi-data7/miyanishi/Data/3RScan/3RScan.json'
    rescan2ref = read_transform_matrix(META_FILE)
    
    batch_export()
