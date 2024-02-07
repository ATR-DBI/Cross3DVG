import os
import numpy as np
from plyfile import PlyData, PlyElement
import pandas as pd

from scannet200_constants import *

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

def read_plymesh(filepath):
    """Read ply file and return it as numpy array. Returns None if emtpy."""
    with open(filepath, 'rb') as f:
        plydata = PlyData.read(f)
    if plydata.elements:
        vertices = pd.DataFrame(plydata['vertex'].data).values
        faces = np.array([f[0] for f in plydata["face"].data])
        return vertices, faces

def save_plymesh(vertices, faces, filename, verbose=True, with_label=True):
    """Save an RGB point cloud as a PLY file.

    Args:
      points_3d: Nx6 matrix where points_3d[:, :3] are the XYZ coordinates and points_3d[:, 4:] are
          the RGB values. If Nx3 matrix, save all points with [128, 128, 128] (gray) color.
    """
    assert vertices.ndim == 2
    if with_label:
        if vertices.shape[1] == 7:
            python_types = (float, float, float, int, int, int, int)
            npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                         ('blue', 'u1'), ('label', 'u4')]

        if vertices.shape[1] == 8:
            python_types = (float, float, float, int, int, int, int, int)
            npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                         ('blue', 'u1'), ('label', 'u4'), ('instance_id', 'u4')]

        if vertices.shape[1] == 10:
            python_types = (float, float, float, int, int, int, float, float, float, int)
            npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                         ('blue', 'u1'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'), ('label', 'u4')]
                        
        if vertices.shape[1] == 11:
            python_types = (float, float, float, int, int, int, float, float, float, int, int)
            npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                         ('blue', 'u1'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'), ('label', 'u4'), ('instance_id', 'u4')]            
    else:
        if vertices.shape[1] == 3:
            gray_concat = np.tile(np.array([128], dtype=np.uint8), (vertices.shape[0], 3))
            vertices = np.hstack((vertices, gray_concat))
        elif vertices.shape[1] == 6:
            python_types = (float, float, float, int, int, int)
            npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                         ('blue', 'u1')]
        elif vertices.shape[1] == 9:
            python_types = (float, float, float, int, int, int, float, float, float)
            npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                         ('blue', 'u1'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]            
        else:
            pass

    vertices_list = []
    for row_idx in range(vertices.shape[0]):
        cur_point = vertices[row_idx]
        vertices_list.append(tuple(dtype(point) for dtype, point in zip(python_types, cur_point)))
    vertices_array = np.array(vertices_list, dtype=npy_types)
    elements = [PlyElement.describe(vertices_array, 'vertex')]

    if faces is not None:
        faces_array = np.empty(len(faces), dtype=[('vertex_indices', 'i4', (3,))])
        faces_array['vertex_indices'] = faces
        elements += [PlyElement.describe(faces_array, 'face')]

    # Write
    PlyData(elements).write(filename)

    if verbose is True:
        print('Saved point cloud to: %s' % filename)


# Map the raw category id to the point cloud
def point_indices_from_group(points, aligned_points, seg_indices, group, labels_pd, CLASS_IDs):
    group_segments = np.array(group['segments'])
    label = group['label']
    
    # Map the category name to id
    label_ids = labels_pd[labels_pd['raw_category'] == label]['id']
    label_id = int(label_ids.iloc[0]) if len(label_ids) > 0 else 0

    # Only store for the valid categories
    if not label_id in CLASS_IDs:
        label_id = 0

    # get points, where segment indices (points labelled with segment ids) are in the group segment list
    point_IDs = np.where(np.isin(seg_indices, group_segments))
    
    return points[point_IDs], aligned_points[point_IDs], point_IDs[0], label_id


# Uncomment out if mesh voxelization is required
# import trimesh
# from trimesh.voxel import creation
# from sklearn.neighbors import KDTree
# import MinkowskiEngine as ME


# VOXELIZE the scene from sampling on the mesh directly instead of vertices
def voxelize_pointcloud(points, colors, labels, instances, faces, voxel_size=0.2):

    # voxelize mesh first and determine closest labels with KDTree search
    trimesh_scene_mesh = trimesh.Trimesh(vertices=points, faces=faces)
    voxel_grid = creation.voxelize(trimesh_scene_mesh, voxel_size)
    voxel_cloud = np.asarray(voxel_grid.points)
    orig_tree = KDTree(points, leaf_size=8)
    _, voxel_pc_matches = orig_tree.query(voxel_cloud, k=1)
    voxel_pc_matches = voxel_pc_matches.flatten()

    # match colors to voxel ids
    points = points[voxel_pc_matches] / voxel_size
    colors = colors[voxel_pc_matches]
    labels = labels[voxel_pc_matches]
    instances = instances[voxel_pc_matches]

    # Voxelize scene
    quantized_scene, scene_inds = ME.utils.sparse_quantize(points, return_index=True)
    quantized_scene_colors = colors[scene_inds]
    quantized_labels = labels[scene_inds]
    quantized_instances = instances[scene_inds]
    
    return quantized_scene, quantized_scene_colors, quantized_labels, quantized_instances
