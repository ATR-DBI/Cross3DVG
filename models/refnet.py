import torch
import torch.nn as nn
import numpy as np
import sys
import os

from models.backbone_module import Pointnet2Backbone
from models.voting_module import VotingModule
from models.lang_module import LangModule
from models.match_module import MatchModule

from models.proposal_module import ProposalModule
from models.proposal_module_votenet import ProposalModuleVoteNet


class TransReferNet(nn.Module):
    def __init__(self, args, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
                 input_feature_dim=0, num_proposal=128, vote_factor=1, sampling="vote_fps",
                 use_lang_classifier=True, use_bidir=False, no_reference=False,
                 emb_size=300, lang_hidden_size=256, dataset_config=None):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert (mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling
        self.use_lang_classifier = use_lang_classifier
        self.use_bidir = use_bidir
        self.no_reference = no_reference
        self.dataset_config = dataset_config

        # --------- PROPOSAL GENERATION ---------
        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Hough voting
        self.vgen = VotingModule(self.vote_factor, 256)

        # Vote aggregation and object proposal
        config_transformer = None

        config_transformer = {
            'mask': 'no_mask',
            'weighted_input': True,
            'transformer_type': 'myAdd_20;deformable',
            'deformable_type': 'myAdd',
            'position_embedding': 'none',
            'input_dim': 0,
            'enc_layers': 0,
            'dec_layers': args.det_dec_layers,
            'dim_feedforward': args.det_dim_feedforward,
            'hidden_dim': args.det_hidden_dim, # 288,
            'dropout': 0.1,
            'nheads': 8,
            'pre_norm': False
        }
        
        if hasattr(args, 'proposal_module_votenet') and args.proposal_module_votenet:
            self.proposal = ProposalModuleVoteNet(num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling)
        else:
            self.proposal = ProposalModule(num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal,
                                        sampling, args.proposal_size,
                                        config_transformer=config_transformer, dataset_config=dataset_config)

        if not no_reference:
            # --------- LANGUAGE ENCODING ---------
            # Encode the input descriptions into vectors
            # (including attention and language classification)
            self.lang = LangModule(args, num_class, use_lang_classifier, use_bidir, emb_size, lang_hidden_size, args.hidden_size)

            # --------- PROPOSAL MATCHING ---------
            # Match the generated proposals and select the most confident ones
            self.match = MatchModule(args, num_proposals=num_proposal, lang_size=(1 + int(self.use_bidir)) * lang_hidden_size,
                                    hidden_size=args.hidden_size, det_channel=config_transformer['hidden_dim']) 

    def forward(self, data_dict):
        """ Forward pass of the network

        Args:
            data_dict: dict
                {
                    point_clouds,
                    lang_feat
                }

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """

        #######################################
        #                                     #
        #           DETECTION BRANCH          #
        #                                     #
        #######################################

        # --------- HOUGH VOTING ---------
        data_dict = self.backbone_net(data_dict)

        # --------- HOUGH VOTING ---------
        xyz = data_dict["fp2_xyz"]
        features = data_dict["fp2_features"]
        data_dict["seed_inds"] = data_dict["fp2_inds"]
        data_dict["seed_xyz"] = xyz
        data_dict["seed_features"] = features

        xyz, features = self.vgen(xyz, features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        data_dict["vote_xyz"] = xyz
        data_dict["vote_features"] = features

        # --------- PROPOSAL GENERATION ---------
        data_dict = self.proposal(xyz, features, data_dict)

        if not self.no_reference:
            #######################################
            #                                     #
            #           LANGUAGE BRANCH           #
            #                                     #
            #######################################

            # --------- LANGUAGE ENCODING ---------
            data_dict = self.lang(data_dict)

            #######################################
            #                                     #
            #          PROPOSAL MATCHING          #
            #                                     #
            #######################################

            # --------- PROPOSAL MATCHING ---------
            # config for bbox_embedding
            data_dict = self.match(data_dict)

        return data_dict
