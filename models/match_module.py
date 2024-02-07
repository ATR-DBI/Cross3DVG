import torch
import torch.nn as nn
from models.transformer.attention import MultiHeadAttention
import random
from models.utils import get_image_feature_around_box


class MatchModule(nn.Module):
    def __init__(self, args, num_proposals=256, lang_size=256, hidden_size=128, lang_num_size=300, det_channel=288*4, head=4, depth=2):
        super().__init__()
        self.args = args

        if args.dataset == 'scanrefer':
            from lib.scanrefer_config import CONF
        elif 'riorefer' in args.dataset:
            from lib.riorefer_config import CONF
        else:
            raise NotImplementedError            
        
        self.use_dist_weight_matrix = True 
        self.num_proposals = num_proposals # 128
        self.lang_size = lang_size
        self.hidden_size = hidden_size # 256
        self.depth = depth - 1 # 1
        
        if 'ViT' in self.args.clip_model:
            clip_size = 512
        elif 'RN50' == self.args.clip_model:
            clip_size = 1024
        elif 'RN50x4' == self.args.clip_model:
            clip_size = 640                
        else:
            raise NotImplemented
        self.clip_size = clip_size
        
        if self.args.use_image_clip:
            self.proj_image = nn.Sequential(
                nn.Linear(clip_size, hidden_size), 
                nn.LayerNorm(hidden_size)
            )
        
        self.features_concat = nn.Sequential(
            nn.Conv1d(det_channel, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, 1),
        )

        self.match = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Conv1d(hidden_size, 1, 1)
        )
                      
        if self.args.match_joint_score:
            self.match_2d = nn.Sequential(
                nn.Conv1d(hidden_size, hidden_size, 1),
                nn.BatchNorm1d(hidden_size),
                nn.PReLU(),
                nn.Conv1d(hidden_size, hidden_size, 1),
                nn.BatchNorm1d(hidden_size),
                nn.PReLU(),
                nn.Conv1d(hidden_size, 1, 1)
            )  
            self.match_3d = nn.Sequential(
                nn.Conv1d(hidden_size, hidden_size, 1),
                nn.BatchNorm1d(hidden_size),
                nn.PReLU(),
                nn.Conv1d(hidden_size, hidden_size, 1),
                nn.BatchNorm1d(hidden_size),
                nn.PReLU(),
                nn.Conv1d(hidden_size, 1, 1)
            )
        self.self_attn = nn.ModuleList(
            MultiHeadAttention(d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head) for i in range(depth))
        self.cross_attn = nn.ModuleList(
            MultiHeadAttention(d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head) for i in range(depth))  # k, q, v
        
    def forward(self, data_dict):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """
        if self.use_dist_weight_matrix:
            # Attention Weight
            objects_center = data_dict['center']
            N_K = objects_center.shape[1]
            center_A = objects_center[:, None, :, :].repeat(1, N_K, 1, 1)
            center_B = objects_center[:, :, None, :].repeat(1, 1, N_K, 1)
            dist = (center_A - center_B).pow(2)
            # print(dist.shape, '<< dist shape', flush=True)
            dist = torch.sqrt(torch.sum(dist, dim=-1))[:, None, :, :]
            dist_weights = 1 / (dist+1e-2)
            norm = torch.sum(dist_weights, dim=2, keepdim=True)
            dist_weights = dist_weights / norm
            zeros = torch.zeros_like(dist_weights)
            # slightly different with our ICCV paper, which leads to higher results (3DVG-Transformer+)
            dist_weights = torch.cat([dist_weights, -dist, zeros, zeros], dim=1).detach()
            attention_matrix_way = 'add'
        else:
            dist_weights = None
            attention_matrix_way = 'mul'
        
        if hasattr(self.args, 'proposal_module_votenet') and self.args.proposal_module_votenet:
            features_3d = data_dict['aggregated_vote_features'] # batch_size, num_proposal, 128
        else:
            # object proposal embedding
            features = data_dict['detr_features'] # 8, 256, 288
            # features = features.reshape(B, N, -1).permute(0, 2, 1)
            features = features.permute(0, 2, 1)
            # features_3d: batch, num_proposal, hidden (8, 256, 128)
            features_3d = self.features_concat(features).permute(0, 2, 1)   
        batch_size, num_proposal = features_3d.shape[:2]            
        
        objectness_masks = data_dict['objectness_scores'].max(2)[1].float().unsqueeze(2)  # batch_size, num_proposals, 1
        features_3d = self.self_attn[0](features_3d, features_3d, features_3d, attention_weights=dist_weights, way=attention_matrix_way)

        # lang_fea: batch_size * len_num_max, num_words, hidden_size (256, 81, 128)        
        lang_fea = data_dict["lang_fea"]
        len_nun_max = data_dict["lang_len_list"].shape[1]
        data_dict["random"] = random.random()

        if self.args.use_text_clip:
            # text_feats: batch, lang_num_max, clip_size (8, 32, 512)
            text_feats = data_dict["ref_text_feat"].detach()
        else:
            text_feats = None
            
        if dist_weights is not None:
            dist_weights = dist_weights[:, None, :, :, :].repeat(1, len_nun_max, 1, 1, 1).reshape(batch_size*len_nun_max, dist_weights.shape[1], num_proposal, num_proposal)
            
        # get clip features of images around the box
        if self.args.use_image_clip:
            # data_dict["center"]: batch, num_proposal (K), 3 (8, 256, 3)
            # data_dict['center_label']: batch, K2, 3 (8, 128, 3)
            # data_dict["image_feat"]: batch, MAX_NUM_FRAME(420), 512
            # data_dict["frame_id"]: batch, MAX_NUM_FRAME(420) 
            # data_dict["camera_pose"]: batch, MAX_NUM_FRAME(420), 4, 4
            # data_dict["num_frame"]: batch
            #
            # img_feats: batch, num_proposal, num_frame, clip_size (8, 256, 10, 512)
            # frame_ids: batch, num_box, num_frame (8, 256, 10)
            frame_ids, img_feats, img_mask = get_image_feature_around_box(data_dict["center"], 
                                    data_dict["camera_poses"], data_dict["num_frame"], 
                                    data_dict["image_feats"], data_dict["frame_id"],
                                    self.args.num_image_clip, self.args.image_clip_cossim)
            data_dict['pred_box_frame_ids'] = frame_ids

            num_frame, clip_size = img_feats.shape[2:]
            # text_feats_det: batch, lang_num_max, 1, 1, clip_size (8, 32, 1, 1, 512)
            if text_feats is not None:
                text_feats = text_feats[:, :, None, None, :]
            # img_feats: batch, lang_num_max, num_proposal, num_frame, clip_size (8, 32, 256, 10, 512)
            img_feats = img_feats[:, None, :, :, :].expand(batch_size, len_nun_max, num_proposal, num_frame, clip_size)
            img_feats = ((img_feats * text_feats).sum(-1).unsqueeze(-1) * img_feats).sum(-2)
            # img_feats: batch * lang_num_max, num_proposal, clip_size
            img_feats = img_feats.view(batch_size * len_nun_max, num_proposal, -1)
            # features_2d: batch * lang_num_max, num_proposal, hidden_size
            features_2d = self.proj_image(img_feats)                
                    
        # feature0_3d: batch, num_proposal, hidden (8, 256, 128)
        feature0_3d = self._copy_paste(features_3d, objectness_masks, batch_size, num_proposal, data_dict)
        # feature1: batch * lang_num_max, num_proposal, hidden (256, 256, 128)         
        feature1 = feature0_3d[:, None, :, :].repeat(1, len_nun_max, 1, 1).reshape(batch_size*len_nun_max, num_proposal, -1)
        feature1 = self.cross_attn[0](feature1, lang_fea, lang_fea, data_dict["attention_mask"])

        for _ in range(self.depth):
            feature1 = self.self_attn[_+1](feature1, feature1, feature1, attention_weights=dist_weights, way=attention_matrix_way)
            feature1 = self.cross_attn[_+1](feature1, lang_fea, lang_fea, data_dict["attention_mask"])

        # fuse 2d 3d features
        feature1_agg = self._fuse_2d3d(feature1, features_2d)
        # feature1_agg: batch_size * len_nun_max, hidden, num_proposals, (256, 128, 256)        
        feature1_agg = feature1_agg.permute(0, 2, 1).contiguous()
        # confidence: batch_size * len_nun_max, num_proposals            
        confidence = self.match(feature1_agg).squeeze(1)  
        data_dict["cluster_ref"] = confidence
            
        if self.args.match_joint_score:
            # features_2d: batch_size * len_nun_max, hidden, num_proposals, (256, 128, 256)         
            features_2d = features_2d.permute(0, 2, 1).contiguous()
            # features_3d: batch_size * len_nun_max, hidden, num_proposals, (256, 128, 256)         
            features_3d = feature1.permute(0, 2, 1).contiguous()            
            # confidence_aux: batch_size * len_nun_max, num_proposals            
            data_dict["cluster_ref_2d"] = self.match_2d(features_2d).squeeze(1)  
            data_dict["cluster_ref_3d"] = self.match_3d(features_3d).squeeze(1)  
            
        return data_dict
    
    def _fuse_2d3d(self, feature1, features_2d):
        return feature1 + features_2d
    
    def _copy_paste(self, features, objectness_masks, batch_size, num_proposal, data_dict):
        # copy paste
        feature0 = features.clone()
        if data_dict["istrain"][0] == 1 and data_dict["random"] < 0.5:
            obj_masks = objectness_masks.bool().squeeze(2)  # batch_size, num_proposals
            obj_lens = torch.zeros(batch_size, dtype=torch.int).cuda()
            for i in range(batch_size):
                obj_mask = torch.where(obj_masks[i, :] == True)[0]
                obj_len = obj_mask.shape[0]
                obj_lens[i] = obj_len

            obj_masks_reshape = obj_masks.reshape(batch_size*num_proposal)
            obj_features = features.reshape(batch_size*num_proposal, -1)
            obj_mask = torch.where(obj_masks_reshape[:] == True)[0]
            total_len = obj_mask.shape[0]
            obj_features = obj_features[obj_mask, :].repeat(2,1)  # total_len, hidden_size
            j = 0
            for i in range(batch_size):
                obj_mask = torch.where(obj_masks[i, :] == False)[0]
                obj_len = obj_mask.shape[0]
                j += obj_lens[i]
                if obj_len < total_len - obj_lens[i]:
                    feature0[i, obj_mask, :] = obj_features[j:j + obj_len, :]
                else:
                    feature0[i, obj_mask[:total_len - obj_lens[i]], :] = obj_features[j:j + total_len - obj_lens[i], :]
        return feature0
