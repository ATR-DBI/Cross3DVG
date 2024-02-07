""" 
Modified from: https://github.com/facebookresearch/votenet/blob/master/scannet/model_util_scannet.py
"""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), os.pardir, "lib")) # HACK add the lib folder
sys.path.append(os.path.join(os.getcwd(), os.pardir, "data/scannet200/"))

from lib.scanrefer_config import CONF
from utils.box_util import get_3d_box

from data.scannet.scannet200_constants import CLASS_LABELS_20, CLASS_LABELS_200, VALID_CLASS_IDS_200
from data.scannet.scannet200_splits import HEAD_CATS_SCANNET_200, COMMON_CATS_SCANNET_200, TAIL_CATS_SCANNET_200
import pandas as pd

#SCANNET_V2_TSV = os.path.join(CONF.PATH.SCANNET_META, "scannetv2-labels.combined.tsv")


def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds

def rotate_aligned_boxes(input_boxes, rot_mat):    
    centers, lengths = input_boxes[:,0:3], input_boxes[:,3:6]    
    new_centers = np.dot(centers, np.transpose(rot_mat))
           
    dx, dy = lengths[:,0]/2.0, lengths[:,1]/2.0
    new_x = np.zeros((dx.shape[0], 4))
    new_y = np.zeros((dx.shape[0], 4))
    
    for i, crnr in enumerate([(-1,-1), (1, -1), (1, 1), (-1, 1)]):        
        crnrs = np.zeros((dx.shape[0], 3))
        crnrs[:,0] = crnr[0]*dx
        crnrs[:,1] = crnr[1]*dy
        crnrs = np.dot(crnrs, np.transpose(rot_mat))
        new_x[:,i] = crnrs[:,0]
        new_y[:,i] = crnrs[:,1]
    
    new_dx = 2.0*np.max(new_x, 1)
    new_dy = 2.0*np.max(new_y, 1)    
    new_lengths = np.stack((new_dx, new_dy, lengths[:,2]), axis=1)
                  
    return np.concatenate([new_centers, new_lengths], axis=1)

def rotate_aligned_boxes_along_axis(input_boxes, rot_mat, axis):    
    centers, lengths = input_boxes[:,0:3], input_boxes[:,3:6]    
    new_centers = np.dot(centers, np.transpose(rot_mat))

    if axis == "x":     
        d1, d2 = lengths[:,1]/2.0, lengths[:,2]/2.0
    elif axis == "y":
        d1, d2 = lengths[:,0]/2.0, lengths[:,2]/2.0
    else:
        d1, d2 = lengths[:,0]/2.0, lengths[:,1]/2.0

    new_1 = np.zeros((d1.shape[0], 4))
    new_2 = np.zeros((d1.shape[0], 4))
    
    for i, crnr in enumerate([(-1,-1), (1, -1), (1, 1), (-1, 1)]):        
        crnrs = np.zeros((d1.shape[0], 3))
        crnrs[:,0] = crnr[0]*d1
        crnrs[:,1] = crnr[1]*d2
        crnrs = np.dot(crnrs, np.transpose(rot_mat))
        new_1[:,i] = crnrs[:,0]
        new_2[:,i] = crnrs[:,1]
    
    new_d1 = 2.0*np.max(new_1, 1)
    new_d2 = 2.0*np.max(new_2, 1)    

    if axis == "x":     
        new_lengths = np.stack((lengths[:,0], new_d1, new_d2), axis=1)
    elif axis == "y":
        new_lengths = np.stack((new_d1, lengths[:,1], new_d2), axis=1)
    else:
        new_lengths = np.stack((new_d1, new_d2, lengths[:,2]), axis=1)
    return np.concatenate([new_centers, new_lengths], axis=1)


class ScannetDatasetConfig(object):
    def __init__(self, labelset='head'):

        #use_labels = ['cabinet', 'bed', 'chair', 'sofa chair', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub'] # otherfurnitureは除く
        if labelset == 'head':
            use_labels = HEAD_CATS_SCANNET_200
        elif labelset == 'head_common':            
            use_labels = HEAD_CATS_SCANNET_200 + COMMON_CATS_SCANNET_200
        elif labelset == 'head_common_tail':         
            use_labels = HEAD_CATS_SCANNET_200 + COMMON_CATS_SCANNET_200 + TAIL_CATS_SCANNET_200       
        else:
            raise NotImplementedError

        # 壁・床・天井は除去するかどうか（ScanReferは除く）
        #remove_surfaces = False
        remove_surfaces = True
        if remove_surfaces:
            surface_objects = set(['floor', 'floor /other room', 'ceiling', 'ceiling /other room', 'wall', 'wall /other room'])
            use_labels = [label for label in use_labels if label not in surface_objects]
            
        def get_scannet200_labelmap():
            labelmap = {label_name:label_id for label_name, label_id in zip(CLASS_LABELS_200, VALID_CLASS_IDS_200)}
            return labelmap

        # label_idとlabel_nameのマップ
        scannet200_labelmap = get_scannet200_labelmap()            

        # class_num: classificationで使うclassの番号0〜N
        # label_name: raw_category名            
        self.type2class = {label_name:class_num for class_num, label_name in enumerate(use_labels)}
        # label_idsは、scannetv2-labels.combinedのraw_categoryに対応するid。use_labelsに該当するものだけを使用
        self.label_ids = np.array([scannet200_labelmap[label_name] for label_name in self.type2class.keys()])
        self.type2class['others'] = len(self.type2class) # 該当しない物体用
        #print(self.type2class)
        #exit()        
        self.class2type = {self.type2class[t]:t for t in self.type2class}
        # label_id2classは、cannetv2-labels.combinedのidをclassification用の番号に変換        
        # label_nameでtype2classにないものはtype2class["others"]
        self.label_id2class = {scannet200_labelmap[label_name]:self.type2class.get(label_name, self.type2class['others']) for label_name in scannet200_labelmap.keys()} 
        self.label_id2class[0] = self.type2class['others'] # 前処理でScanNet200以外のラベルは0にしたので        

        self.num_class = len(self.type2class.keys())
        self.num_heading_bin = 1
        self.num_size_cluster = len(self.type2class.keys())
        
        # bboxの平均のサイズ, mean_size_arrのindexがclass_indに対応
        #bbox_size_df = pd.read_pickle(os.path.join(CONF.PATH.META, 'scannet_reference_bbox_size.pkl'))
        #bbox_size_df = pd.read_pickle(os.path.join(CONF.PATH.META, 'rio_bbox_size.pkl'))
        bbox_size_df = pd.read_pickle(os.path.join(CONF.PATH.META, 'bbox_size.pkl'))
        bbox_size_df['class_ind'] = bbox_size_df.label_id.map(self.label_id2class)
        self.mean_size_arr = np.array(bbox_size_df.groupby('class_ind').mean().reset_index().sort_values(['class_ind'])[['x_len', 'y_len', 'z_len']])
        
        self.type_mean_size = {}
        for class_ind in range(self.num_size_cluster):
            self.type_mean_size[self.class2type[class_ind]] = self.mean_size_arr[class_ind,:]        


    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class
            [optinal] also small regression number from  
            class center angle to current angle.
            angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
            return is class of int32 of 0,1,...,N-1 and a number such that
                class*(2pi/N) + number = angle

            NOT USED.
        '''
        assert(False)
    
    def class2angle(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class.
        
        As ScanNet only has axis-alined boxes so angles are always 0. '''
        return 0

    def class2angle_batch(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class.
        
        As ScanNet only has axis-alined boxes so angles are always 0. '''
        return np.zeros(pred_cls.shape[0])

    def size2class(self, size, type_name):
        ''' Convert 3D box size (l,w,h) to size class and size residual '''
        size_class = self.type2class[type_name]
        size_residual = size - self.type_mean_size[type_name]
        return size_class, size_residual
    
    def class2size(self, pred_cls, residual):
        ''' Inverse function to size2class '''      
        return self.mean_size_arr[pred_cls] + residual

    def class2size_batch(self, pred_cls, residual):
        ''' Inverse function to size2class '''      
        return self.mean_size_arr[pred_cls] + residual

    def param2obb(self, center, heading_class, heading_residual, size_class, size_residual):
        heading_angle = self.class2angle(heading_class, heading_residual)
        box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle*-1
        return obb

    def param2obb_batch(self, center, heading_class, heading_residual, size_class, size_residual):
        heading_angle = self.class2angle_batch(heading_class, heading_residual)
        box_size = self.class2size_batch(size_class, size_residual)
        obb = np.zeros((heading_class.shape[0], 7))
        obb[:, 0:3] = center
        obb[:, 3:6] = box_size
        obb[:, 6] = heading_angle*-1
        return obb    
