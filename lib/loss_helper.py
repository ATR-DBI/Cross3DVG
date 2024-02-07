# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.nn_distance import nn_distance, huber_loss
from lib.ap_helper import parse_predictions
from lib.loss import SoftmaxRankingLoss 
from utils.box_util import get_3d_box, get_3d_box_batch, box3d_iou, box3d_iou_batch

# FAR_THRESHOLD = 0.6
FAR_THRESHOLD = 0.3
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3  # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2, 0.8]  # put larger weights on positive objectness


def compute_vote_loss(data_dict):
    """ Compute vote loss: Match predicted votes to GT votes.

    Args:
        data_dict: dict (read-only)

    Returns:
        vote_loss: scalar Tensor

    Overall idea:
        If the seed point belongs to an object (votes_label_mask == 1),
        then we require it to vote for the object center.

        Each seed point may vote for multiple translations v1,v2,v3
        A seed point may also be in the boxes of multiple objects:
        o1,o2,o3 with corresponding GT votes c1,c2,c3

        Then the loss for this seed point is:
            min(d(v_i,c_j)) for i=1,2,3 and j=1,2,3
    """
    # Load ground truth votes and assign them to seed points
    batch_size = data_dict['seed_xyz'].shape[0]
    num_seed = data_dict['seed_xyz'].shape[1]  # B,num_seed,3
    vote_xyz = data_dict['vote_xyz']  # B,num_seed*vote_factor,3
    seed_inds = data_dict['seed_inds'].long()  # B,num_seed in [0,num_points-1]

    # Get groundtruth votes for the seed points
    # vote_label_mask: Use gather to select B,num_seed from B,num_point
    #   non-object point has no GT vote mask = 0, object point has mask = 1
    # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
    #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
    seed_gt_votes_mask = torch.gather(data_dict['vote_label_mask'], 1, seed_inds)
    seed_inds_expand = seed_inds.view(batch_size, num_seed, 1).repeat(1, 1, 3 * GT_VOTE_FACTOR)
    seed_gt_votes = torch.gather(data_dict['vote_label'], 1, seed_inds_expand)
    seed_gt_votes += data_dict['seed_xyz'].repeat(1, 1, 3)

    # Compute the min of min of distance
    vote_xyz_reshape = vote_xyz.view(batch_size * num_seed, -1, 3)  # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_gt_votes_reshape = seed_gt_votes.view(batch_size * num_seed, GT_VOTE_FACTOR, 3)  # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
    votes_dist, _ = torch.min(dist2, dim=1)  # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist = votes_dist.view(batch_size, num_seed)
    vote_loss = torch.sum(votes_dist * seed_gt_votes_mask.float()) / (torch.sum(seed_gt_votes_mask.float()) + 1e-6)

    return vote_loss


def compute_objectness_loss(data_dict):
    """ Compute objectness loss for the proposals.

    Args:
        data_dict: dict (read-only)

    Returns:
        objectness_loss: scalar Tensor
        objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        objectness_mask: (batch_size, num_seed) Tensor with value 0 or 1
        object_assignment: (batch_size, num_seed) Tensor with long int
            within [0,num_gt_object-1]
    """
    # Associate proposal and GT objects by point-to-point distances
    aggregated_vote_xyz = data_dict['aggregated_vote_xyz']
    gt_center = data_dict['center_label'][:, :, 0:3]
    B = gt_center.shape[0]
    K = aggregated_vote_xyz.shape[1] # num_proposal
    K2 = gt_center.shape[1] # num_object_label
    dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center)  # dist1: BxK, dist2: BxK2

    # Generate objectness label and mask
    # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
    # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
    euclidean_dist1 = torch.sqrt(dist1 + 1e-6)
    objectness_label = torch.zeros((B, K), dtype=torch.long).cuda()
    objectness_mask = torch.zeros((B, K)).cuda()
    objectness_label[euclidean_dist1 < NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1 < NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1 > FAR_THRESHOLD] = 1
    # Compute objectness loss
    objectness_scores = data_dict['objectness_scores'] # batch, 
    criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
    objectness_loss = criterion(objectness_scores.transpose(2, 1), objectness_label)
    objectness_loss = torch.sum(objectness_loss * objectness_mask) / (torch.sum(objectness_mask) + 1e-6)

    # Set assignment
    object_assignment = ind1  # (B,K) with values in 0,1,...,K2-1

    return objectness_loss, objectness_label, objectness_mask, object_assignment


def compute_box_and_sem_cls_loss(data_dict, config):
    """ Compute 3D bounding box and semantic classification loss.

    Args:
        data_dict: dict (read-only)

    Returns:
        center_loss
        heading_cls_loss
        heading_reg_loss
        size_cls_loss
        size_reg_loss
        sem_cls_loss
    """

    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr

    object_assignment = data_dict['object_assignment']
    batch_size = object_assignment.shape[0]
    
    # Compute center loss
    pred_center = data_dict['center']
    gt_center = data_dict['center_label'][:, :, 0:3]
    # dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center) # dist1: BxK, dist2: BxK2
    _, ind1, dist2, _ = nn_distance(pred_center, gt_center)  # dist1: BxK, dist2: BxK2
    box_label_mask = data_dict['box_label_mask']
    objectness_label = data_dict['objectness_label'].float()
    # USE GT_CENTER TO CALCULATE CENTER
    gt_center_label = torch.gather(gt_center, dim=1, index=object_assignment.unsqueeze(-1).repeat(1, 1, 3))
    dist1 = torch.norm(gt_center_label - pred_center, p=2, dim=2)
    dist2 = torch.sqrt(dist2 + 1e-8)
    centroid_reg_loss1 = \
        torch.sum(dist1 * objectness_label) / (torch.sum(objectness_label) + 1e-6)
    centroid_reg_loss2 = \
        torch.sum(dist2 * box_label_mask) / (torch.sum(box_label_mask) + 1e-6)
    center_loss = centroid_reg_loss1 + centroid_reg_loss2
    # Compute heading loss
    heading_class_label = torch.gather(data_dict['heading_class_label'], 1,
                                       object_assignment)  # select (B,K) from (B,K2)
    criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
    heading_class_loss = criterion_heading_class(data_dict['heading_scores'].transpose(2, 1),
                                                 heading_class_label)  # (B,K)
    heading_class_loss = torch.sum(heading_class_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

    heading_residual_label = torch.gather(data_dict['heading_residual_label'], 1,
                                          object_assignment)  # select (B,K) from (B,K2)
    heading_residual_normalized_label = heading_residual_label / (np.pi / num_heading_bin)

    # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
    heading_label_one_hot = torch.cuda.FloatTensor(batch_size, heading_class_label.shape[1], num_heading_bin).zero_()
    heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1), 1)  # src==1 so it's *one-hot* (B,K,num_heading_bin)
    heading_residual_normalized_loss = huber_loss(
        torch.sum(data_dict['heading_residuals_normalized'] * heading_label_one_hot,
                  -1) - heading_residual_normalized_label, delta=1.0)  # (B,K)
    heading_residual_normalized_loss = torch.sum(heading_residual_normalized_loss * objectness_label) / (
                torch.sum(objectness_label) + 1e-6)

    size_class_label = torch.gather(data_dict['size_class_label'], 1, object_assignment)  # select (B,K) from (B,K2)
    criterion_size_class = nn.CrossEntropyLoss(reduction='none')
    size_class_loss = criterion_size_class(data_dict['size_scores'].transpose(2, 1), size_class_label)  # (B,K)
    size_class_loss = torch.sum(size_class_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

    size_residual_label = torch.gather(data_dict['size_residual_label'], 1,
                                       object_assignment.unsqueeze(-1).repeat(1, 1, 3))  # select (B,K,3) from (B,K2,3)
   
    size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], num_size_cluster).zero_()
    size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1), 1) 
    size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1, 1, 1, 3)  # (B,K,num_size_cluster,3)
    predicted_size_residual_normalized = torch.sum(data_dict['size_residuals_normalized'] * size_label_one_hot_tiled, 2)  # (B,K,3)
    mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)  # (1,1,num_size_cluster,3)
    mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2)  # (B,K,3)
    size_residual_label_normalized = size_residual_label / mean_size_label  # (B,K,3)
    size_residual_normalized_loss = torch.mean(
        huber_loss(predicted_size_residual_normalized - size_residual_label_normalized, delta=1.0), -1)  # (B,K,3) -> (B,K)
    size_residual_normalized_loss = torch.sum(size_residual_normalized_loss * objectness_label) / (
                torch.sum(objectness_label) + 1e-6)
    # Semantic cls loss
    sem_cls_label = torch.gather(data_dict['sem_cls_label'], 1, object_assignment)  # select (B,K) from (B,K2)
    criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
    sem_cls_loss = criterion_sem_cls(data_dict['sem_cls_scores'].transpose(2, 1), sem_cls_label)  # (B,K)
    sem_cls_loss = torch.sum(sem_cls_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

    return center_loss, heading_class_loss, heading_residual_normalized_loss, size_class_loss, size_residual_normalized_loss, sem_cls_loss


# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: torch.Tensor, reduction: str) -> torch.Tensor:
    return F.cross_entropy(logits, torch.arange(len(logits), device=logits.device), reduction=reduction)    


def clip_loss(similarity: torch.Tensor, reduction: str) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity, reduction)
    image_loss = contrastive_loss(similarity.T, reduction)
    return (caption_loss + image_loss) / 2.0


def compute_correspond_loss(data_dict):
    # objectness_label: batch, num_proposal (8, 256)
    objectness_label = data_dict['objectness_label'].float()
    # feature_3d: batch, num_proposal, hidden (8, 256, 128)
    feature_3d = data_dict['feature0_3d']
    feature_2d = data_dict['feature0_2d']

    batch_size = feature_3d.shape[0]
    total_loss = 0.     
    for i in range(batch_size):
        # logits_per_3d: num_proposal, num_proposal
        logits_per_3d = torch.matmul(feature_3d[i], feature_2d[i].t()) # * objectness_label[i])
        # num_proposal
        loss = clip_loss(logits_per_3d, reduction='none') * objectness_label[i]
        total_loss = total_loss + torch.sum(loss) / (torch.sum(objectness_label[i]) + 1e-6)        
    return total_loss / batch_size


def compute_clip_fuse_loss(data_dict):
    # object_assignment: batch, num_proposal
    object_assignment = data_dict['object_assignment']
    # objectness_label: batch, num_proposal (8, 256)
    objectness_label = data_dict['objectness_label'].float() # .unsqueeze(-1)
    text_feats = data_dict['obj_text_feats']
    image_feats = data_dict['obj_image_feats']
    batch_size = object_assignment.shape[0]
    batch_chunk_size, num_proposal, hidden_size = image_feats.shape
    chunk_size = int(batch_chunk_size / batch_size)
    object_assignment = object_assignment[:, None, :].repeat(1, chunk_size, 1).reshape(batch_chunk_size, num_proposal)
    objectness_label = objectness_label[:, None, :].repeat(1, chunk_size, 1).reshape(batch_chunk_size, num_proposal)
    image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
    text_feats = torch.gather(text_feats, 1, object_assignment[:, :, None].repeat(1, 1, hidden_size))  # select (B,K) from (B,K2)
    # compute cosine similarity
    loss = 1 - torch.sum(((text_feats * image_feats).sum(dim=-1) * objectness_label)) / (torch.sum(objectness_label) + 1e-6)
    return loss


def compute_det_clip_loss(data_dict, args, tau=1.):
    """ Compute clip loss
    Args:
        data_dict: dict (read-only)

    Returns:
        clip_loss
    """
    assert args.use_text_clip_norm
    
    if 'ViT' in args.clip_model:
        clip_size = 512
    elif 'RN50' == args.clip_model:
        clip_size = 1024
    elif 'RN50x4' == args.clip_model:
        clip_size = 640                
    else:
        raise NotImplemented
    
    # object_assignment: batch, K (num_proposal)
    object_assignment = data_dict['object_assignment']
    batch_size, num_proposal = object_assignment.shape
    # objectness_label: batch, K (8, 256)
    objectness_label = data_dict['objectness_label'].float()
    clip_visual_feat = data_dict["clip_det_visual_feat"] 
    _, num_proposals, clip_size = clip_visual_feat.shape
    clip_visual_feat = clip_visual_feat.view(batch_size, -1, num_proposals, clip_size).contiguous()
    lang_num_max = clip_visual_feat.shape[1]  
    objectness_label = objectness_label[:, None, :].expand(batch_size, lang_num_max, num_proposal)    
    clip_text_feat = data_dict["object_name_feat"]
    clip_text_feat = torch.gather(clip_text_feat, 1, object_assignment[:, :, None].expand(batch_size, num_proposal, clip_size))  # select (B,K,512) from (B,K2,512)
    clip_text_feat = clip_text_feat[:, None, :, :].expand(batch_size, lang_num_max, num_proposal, clip_size).detach() # .reshape(batch_size, num_proposal)
    loss = 1  - torch.sum((clip_text_feat * clip_visual_feat).sum(dim=-1) * objectness_label) / (torch.sum(objectness_label) + 1e-6)

    return loss    


def compute_ref_clip_loss(data_dict, args):
    """ Compute clip loss
    Args:
        data_dict: dict (read-only)

    Returns:
        clip_loss
    """
    assert args.use_text_clip_norm
    
    if 'ViT' in args.clip_model:
        clip_size = 512
    elif 'RN50' == args.clip_model:
        clip_size = 1024
    elif 'RN50x4' == args.clip_model:
        clip_size = 640                
    else:
        raise NotImplemented    

    # object_assignment: batch, num_proposal
    object_assignment = data_dict['object_assignment']
    batch_size = object_assignment.shape[0]
    # objectness_label: batch, num_proposal
    # 1 if pred object center is within NEAR_THRESHOLD of any GT object
    objectness_label = data_dict['objectness_label'].float()
    clip_visual_feat = data_dict["clip_ref_visual_feat"]
    _, num_proposals, clip_size = clip_visual_feat.shape
    clip_visual_feat = clip_visual_feat.view(batch_size, -1, num_proposals, clip_size).contiguous()
    lang_num_max = clip_visual_feat.shape[1]    
    clip_text_feat = data_dict["object_ref_feat"]
    clip_text_feat = torch.gather(clip_text_feat, 1, object_assignment[:, :, None].expand(batch_size, num_proposals, clip_size))  # select (B,K,512) from (B,K2,512)
    clip_text_feat = clip_text_feat[:, None, :, :].expand(batch_size, lang_num_max, num_proposals, clip_size).detach() # .reshape(batch_size, num_proposal)
    objectness_label = objectness_label[:, None, :].expand(batch_size, lang_num_max, num_proposals)
    loss = 1 - torch.sum((clip_text_feat * clip_visual_feat).sum(dim=-1) * objectness_label) / (torch.sum(objectness_label) + 1e-6)

    return loss    



def compute_reference_loss(data_dict, config, args, no_reference=False):
    """ Compute cluster reference loss
    Args:
        data_dict: dict (read-only)

    Returns:
        ref_loss, lang_loss, cluster_preds, cluster_labels
    """
    # unpack
    # cluster_preds = data_dict["cluster_ref"] # (B, num_proposal)

    # predicted bbox
    pred_center = data_dict['center'].detach().cpu().numpy()  # (B,K,3)
    pred_heading_class = torch.argmax(data_dict['heading_scores'], -1)  # B,num_proposal
    pred_heading_residual = torch.gather(data_dict['heading_residuals'], 2,
                                         pred_heading_class.unsqueeze(-1))  # B,num_proposal,1
    pred_heading_class = pred_heading_class.detach().cpu().numpy()  # B,num_proposal
    pred_heading_residual = pred_heading_residual.squeeze(2).detach().cpu().numpy()  # B,num_proposal
    pred_size_class = torch.argmax(data_dict['size_scores'], -1)  # B,num_proposal
    pred_size_residual = torch.gather(data_dict['size_residuals'], 2,
                                      pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1,
                                                                                         3))  # B,num_proposal,1,3
    pred_size_class = pred_size_class.detach().cpu().numpy()
    pred_size_residual = pred_size_residual.squeeze(2).detach().cpu().numpy()  # B,num_proposal,3

    gt_center_list = data_dict['ref_center_label_list'].cpu().numpy()  # (B,3)
    gt_heading_class_list = data_dict['ref_heading_class_label_list'].cpu().numpy()  # B
    gt_heading_residual_list = data_dict['ref_heading_residual_label_list'].cpu().numpy()  # B
    gt_size_class_list = data_dict['ref_size_class_label_list'].cpu().numpy()  # B
    gt_size_residual_list = data_dict['ref_size_residual_label_list'].cpu().numpy()  # B,3
    # convert gt bbox parameters to bbox corners
    batch_size, num_proposals = data_dict['aggregated_vote_features'].shape[:2]
    batch_size, len_nun_max = gt_center_list.shape[:2]
    lang_num = data_dict["lang_num"] 
    max_iou_rate_25 = 0
    max_iou_rate_5 = 0

    if not no_reference:
        cluster_preds = data_dict["cluster_ref"].reshape(batch_size, len_nun_max, num_proposals)
        if args.match_joint_score:
            cluster_preds_2d = data_dict["cluster_ref_2d"].reshape(batch_size, len_nun_max, num_proposals)
            cluster_preds_3d = data_dict["cluster_ref_3d"].reshape(batch_size, len_nun_max, num_proposals)
    else:
        cluster_preds = torch.zeros(batch_size, len_nun_max, num_proposals).cuda()
        
    criterion = SoftmaxRankingLoss()
    loss = 0.
    gt_labels = np.zeros((batch_size, len_nun_max, num_proposals))
    for i in range(batch_size):
        objectness_masks = data_dict['objectness_scores'].max(2)[1].float().cpu().numpy() # batch_size, num_proposals
        gt_obb_batch = config.param2obb_batch(gt_center_list[i][:, 0:3], gt_heading_class_list[i],
                                              gt_heading_residual_list[i],
                                              gt_size_class_list[i], gt_size_residual_list[i])
        gt_bbox_batch = get_3d_box_batch(gt_obb_batch[:, 3:6], gt_obb_batch[:, 6], gt_obb_batch[:, 0:3])
        labels = np.zeros((len_nun_max, num_proposals))
        for j in range(len_nun_max):
            if j < lang_num[i]:
                # convert the bbox parameters to bbox corners
                pred_obb_batch = config.param2obb_batch(pred_center[i, :, 0:3], pred_heading_class[i],
                                                        pred_heading_residual[i],
                                                        pred_size_class[i], pred_size_residual[i])
                pred_bbox_batch = get_3d_box_batch(pred_obb_batch[:, 3:6], pred_obb_batch[:, 6], pred_obb_batch[:, 0:3])
                ious = box3d_iou_batch(pred_bbox_batch, np.tile(gt_bbox_batch[j], (num_proposals, 1, 1)))

                if data_dict["istrain"][0] == 1 and not no_reference and data_dict["random"] < 0.5:
                    ious = ious * objectness_masks[i]

                ious_ind = ious.argmax()
                max_ious = ious[ious_ind]
                if max_ious >= 0.25:
                    labels[j, ious.argmax()] = 1  # treat the bbox with highest iou score as the gt
                    max_iou_rate_25 += 1
                if max_ious >= 0.5:
                    max_iou_rate_5 += 1

        cluster_labels = torch.FloatTensor(labels).cuda()  # B proposals
        gt_labels[i] = labels
        # compute reference loss
        # cluster_preds:  batch_size, len_nun_max, num_proposals
        loss += criterion(cluster_preds[i, :lang_num[i]], cluster_labels[:lang_num[i]].float().clone())
        if args.match_joint_score:
            loss += criterion(cluster_preds_2d[i, :lang_num[i]], cluster_labels[:lang_num[i]].float().clone())
            loss += criterion(cluster_preds_3d[i, :lang_num[i]], cluster_labels[:lang_num[i]].float().clone())

    data_dict['max_iou_rate_0.25'] = max_iou_rate_25 / sum(lang_num.cpu().numpy())
    data_dict['max_iou_rate_0.5'] = max_iou_rate_5 / sum(lang_num.cpu().numpy())

    cluster_labels = torch.FloatTensor(gt_labels).cuda()  # B len_nun_max proposals
    loss = loss / batch_size
    return data_dict, loss, cluster_preds, cluster_labels


def compute_lang_classification_loss(data_dict):
    criterion = torch.nn.CrossEntropyLoss()
    object_cat_list = data_dict["object_cat_list"]
    batch_size, len_nun_max = object_cat_list.shape[:2]
    lang_num = data_dict["lang_num"]
    lang_scores = data_dict["lang_scores"].reshape(batch_size, len_nun_max, -1)
    loss = 0.
    for i in range(batch_size):
        num = lang_num[i]
        loss += criterion(lang_scores[i, :num], object_cat_list[i, :num])
    loss = loss / batch_size

    return loss


def get_loss(args, data_dict, config, detection=True, reference=True, use_lang_classifier=False):
    """ Loss functions

    Args:
        data_dict: dict
        config: dataset config instance
        reference: flag (False/True)
    Returns:
        loss: pytorch scalar tensor
        data_dict: dict
    """

    # Vote loss
    vote_loss = compute_vote_loss(data_dict)
    # Obj loss
    objectness_loss, objectness_label, objectness_mask, object_assignment = compute_objectness_loss(data_dict)
    num_proposal = objectness_label.shape[1]
    total_num_proposal = objectness_label.shape[0] * objectness_label.shape[1]
    data_dict['objectness_label'] = objectness_label
    data_dict['objectness_mask'] = objectness_mask
    data_dict['object_assignment'] = object_assignment
    data_dict['pos_ratio'] = torch.sum(objectness_label.float().cuda()) / float(total_num_proposal)
    data_dict['neg_ratio'] = torch.sum(objectness_mask.float()) / float(total_num_proposal) - data_dict['pos_ratio']

    # Box loss and sem cls loss
    center_loss, heading_cls_loss, heading_reg_loss, size_cls_loss, size_reg_loss, sem_cls_loss = compute_box_and_sem_cls_loss(
        data_dict, config)
    box_loss = center_loss + 0.1 * heading_cls_loss + heading_reg_loss + 0.1 * size_cls_loss + size_reg_loss

    if detection:
        data_dict['vote_loss'] = vote_loss
        data_dict['objectness_loss'] = objectness_loss
        data_dict['center_loss'] = center_loss
        data_dict['heading_cls_loss'] = heading_cls_loss
        data_dict['heading_reg_loss'] = heading_reg_loss
        data_dict['size_cls_loss'] = size_cls_loss
        data_dict['size_reg_loss'] = size_reg_loss
        data_dict['sem_cls_loss'] = sem_cls_loss
        data_dict['box_loss'] = box_loss
    else:
        data_dict['vote_loss'] = torch.zeros(1)[0].cuda()
        data_dict['objectness_loss'] = torch.zeros(1)[0].cuda()
        data_dict['center_loss'] = torch.zeros(1)[0].cuda()
        data_dict['heading_cls_loss'] = torch.zeros(1)[0].cuda()
        data_dict['heading_reg_loss'] = torch.zeros(1)[0].cuda()
        data_dict['size_cls_loss'] = torch.zeros(1)[0].cuda()
        data_dict['size_reg_loss'] = torch.zeros(1)[0].cuda()
        data_dict['sem_cls_loss'] = torch.zeros(1)[0].cuda()
        data_dict['box_loss'] = torch.zeros(1)[0].cuda()

    if reference:
        # Reference loss
        data_dict, ref_loss, _, cluster_labels = compute_reference_loss(data_dict, config, args)
        data_dict["cluster_labels"] = cluster_labels
        data_dict["ref_loss"] = ref_loss
        
    else:
        # # Reference loss
        ref_loss, ref_loss, _, cluster_labels = compute_reference_loss(data_dict, config, args, no_reference=True)
        lang_count = data_dict['ref_center_label_list'].shape[1]
        data_dict["cluster_labels"] = cluster_labels
        data_dict["cluster_ref"] = objectness_label.new_zeros(objectness_label.shape).float().cuda().repeat(lang_count, 1)
        # store
        data_dict["ref_loss"] = torch.zeros(1)[0].cuda()
        
    data_dict["clip_loss"] = torch.zeros(1)[0].cuda()
    data_dict["corr_loss"] = torch.zeros(1)[0].cuda()  

    if reference and use_lang_classifier:
        data_dict["lang_loss"] = compute_lang_classification_loss(data_dict)
    else:
        data_dict["lang_loss"] = torch.zeros(1)[0].cuda()

    # Final loss function
    loss = args.vote_weight * data_dict['vote_loss'] + args.objectness_weight * data_dict['objectness_loss'] \
        + args.box_weight * data_dict['box_loss'] + args.sem_cls_weight * data_dict['sem_cls_loss'] \
        + args.ref_weight * data_dict["ref_loss"] + args.lang_weight * data_dict["lang_loss"] \
        + args.clip_weight * data_dict['clip_loss'] + args.corr_weight * data_dict['corr_loss']
    loss *= 10  # amplify
    data_dict['loss'] = loss

    return loss, data_dict
