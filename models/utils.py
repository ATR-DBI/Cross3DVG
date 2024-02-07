import torch

# Get camera_pose that the bbox is likely to show as much as possible, and get the clip feature of that camera image
def get_image_feature_around_box(bbox_centers_batch, camera_pose_list, num_pose_list, image_feat_list, frame_id_list,
                                max_num_frame=1, cossim_thresh=0.8):
    batch = len(bbox_centers_batch)
    max_num_box = bbox_centers_batch[0].shape[0] # number of bbox candidates

    camera_rotate = []
    camera_trans = []
    bbox_centers = []
    
    for i in range(batch):
        num_pose = num_pose_list[i] 
        # camera_pose: num_pose, 4, 4 -> num_box * num_pose, 4, 4
        camera_pose = camera_pose_list[i,:num_pose].unsqueeze(0).expand(max_num_box, num_pose, 4, 4).reshape(-1, 4, 4)
        camera_rotate.append(camera_pose[:, :3, :3])
        camera_trans.append(camera_pose[:, :3, 3])
        # bbox_centers: num_box, 3 -> num_box * num_pose, 3
        bbox_centers.append(bbox_centers_batch[i].unsqueeze(1).expand(max_num_box, num_pose, 3).reshape(-1, 3))

    # camera_rotate: batch * num_box * num_pose, 3, 3        
    camera_rotate = torch.cat(camera_rotate)
    camera_trans = torch.cat(camera_trans)
    # bbox_centers: batch * num_box * num_pose, 3
    bbox_centers = torch.cat(bbox_centers)

    # trans_dist: batch * num_box * num_pos
    trans_dist = torch.norm(camera_trans - bbox_centers, dim=-1)
    # trans_vec_all: batch * num_box * num_pos, 3
    trans_vec_all = bbox_centers - camera_trans
    # orient_vec_all: batch * num_box * num_pos, 3
    orient_vec_all = camera_rotate[:, :3, 2] 
    
    # compute cosine similarity: batch * num_box * num_pos
    cossim = (orient_vec_all * trans_vec_all).sum(dim=-1) / torch.norm(orient_vec_all, dim=-1) / torch.norm(trans_vec_all, dim=-1)
    # Make sure that the lower limit of trans_closeness is greater than cossim (-1 to +1)
    trans_closeness = 1 + (1 - trans_dist / trans_dist.max())
    # trans_closeness should be greater than cossim_thresh and should not be affected by cossim.    
    trans_closeness[cossim <= cossim_thresh] = 0
    cossim[cossim > cossim_thresh] = 0    
    # score_all : batch * num_box * num_pos
    score_all = cossim + trans_closeness
    
    # image_feat_list: 
    #   batch, MAX_NUM_FRAME(420), clip_size
    #   or 
    #   batch, MAX_NUM_FRAME(420), channel, height, width
    feat_size = list(image_feat_list.shape[2:])
    top_image_feats = torch.zeros([batch, max_num_box, max_num_frame] + feat_size).to(image_feat_list[0].device)
    top_image_mask =  torch.ones(batch, max_num_box, max_num_frame).to(image_feat_list[0].device)
    
    top_frame_list = []
    curr = 0
    # num_box_pose_list: batch
    for i, num_pose in enumerate(num_pose_list):
        num_box_pose = max_num_box * num_pose
        # score: num_box, num_pose
        score = score_all[curr:(curr+num_box_pose)].reshape(max_num_box, num_pose)
        curr += num_box_pose
        topk = min(num_pose, max_num_frame)
        score_topk_value, score_topk_index = torch.topk(score, topk, dim=1)   
        # image_feat: num_box, topk, 512
        image_feat = image_feat_list[i][score_topk_index]
        top_image_feats[i,:,:topk] = image_feat
        top_image_mask[i,:,:topk] = 0
        top_frame_list.append(frame_id_list[i][score_topk_index])
        
    # top_frame_list[i]: num_box, topk (8, 256, 5)
    # top_image_feats: batch, num_box, max_num_frame, clip_size
    return top_frame_list, top_image_feats.detach(), top_image_mask.bool().detach()
