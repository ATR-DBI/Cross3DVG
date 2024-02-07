import torch
import torch.nn as nn
#from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class MatchModule(nn.Module):
    def __init__(self, num_proposals=256, lang_size=256, hidden_size=128, use_match_gru=False):
        super().__init__() 

        self.num_proposals = num_proposals
        self.lang_size = lang_size
        self.hidden_size = hidden_size
        
        self.fuse = nn.Sequential(
            nn.Conv1d(self.lang_size + 128, hidden_size, 1),
            nn.ReLU()
        )
        
        if use_match_gru:
            self.gru = nn.GRU(
                input_size=hidden_size,
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=True
            )
        
        # self.match = nn.Conv1d(hidden_size, 1, 1)
        self.match = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, 1, 1)
        )

    def forward(self, data_dict):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """

        # unpack outputs from detection branch
        features = data_dict['aggregated_vote_features'] # batch_size, num_proposal, 128
        objectness_masks = data_dict['objectness_scores'].max(2)[1].float().unsqueeze(2) # batch_size, num_proposals, 1

        # unpack outputs from language branch
        lang_feat = data_dict["lang_emb"]
        batch_chunk_size = lang_feat.shape[0]
        lang_feat = lang_feat.unsqueeze(1).repeat(1, self.num_proposals, 1) # batch_size * len_nun_max, num_proposals, lang_size
        batch_size, num_proposals, obj_feat_size = features.shape
        len_nun_max = int(batch_chunk_size / batch_size)
        # features: batch_chunk_size, num_proposals, obj_feat_size [256, 256, 128]
        features = features.unsqueeze(1).repeat(1, len_nun_max, 1, 1).reshape(batch_chunk_size, num_proposals, obj_feat_size)
        # objectness_masks: batch_chunk_size, num_proposals, 1: [256, 256, 1]
        objectness_masks = objectness_masks.unsqueeze(1).repeat(1, len_nun_max, 1, 1).reshape(batch_chunk_size, num_proposals, 1)
        # fuse
        features = torch.cat([features, lang_feat], dim=-1) # batch_chunk_size, num_proposals, 128 + lang_size
        features = features.permute(0, 2, 1).contiguous() # batch_chunk_size, 128 + lang_size, num_proposals

        # fuse features
        features = self.fuse(features) # batch_chunk_size, hidden_size, num_proposals
        
        if hasattr(self, 'gru'):
            features = features.permute(0, 2, 1).contiguous() # batch_chunk_size, num_proposals, hidden_size (256, 256, 128)
            features, _ = self.gru(features) # batch_chunk_size, num_proposals, hidden_size * (1 + use_bidir) 256, 256, 256
            # features: batch_chunk_size, hidden_size, num_proposals (256, 128, 256)
            features = (features[:,:,0:self.hidden_size] + features[:,:,self.hidden_size:]).permute(0, 2, 1).contiguous() 
        
        # mask out invalid proposals
        objectness_masks = objectness_masks.permute(0, 2, 1).contiguous() # batch_chunk_size, 1, num_proposals
        features = features * objectness_masks
        
        # match
        confidences = self.match(features).squeeze(1) # batch_chunk_size, num_proposals
        data_dict["cluster_ref"] = confidences
        
        # ScanReferはcompute_reference_loss()内で、ious = ious * objectness_masks[i]していないので
        data_dict["random"] = 1

        return data_dict
