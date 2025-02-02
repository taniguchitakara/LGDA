import torch
import torch.nn as nn

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import DETECTORS, build_roi_extractor, build_head, HEADS
from .two_stage import TwoStageDetector
from ..plugins.match_module import MatchModule
from ..plugins.generate_ref_roi_feats import generate_ref_roi_feats, add_gaussian_noise,add_gaussian_noise_old
from mmcv.cnn import xavier_init
import mmcv
import numpy as np
from mmcv.image import imread, imwrite
from mmcv.runner import (build_optimizer)
import cv2
from mmcv.visualization.color import color_val
from random import  choice
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead
from mmdet.models.roi_heads.test_mixins import BBoxTestMixin, MaskTestMixin
import copy
import torch.optim as optim



@DETECTORS.register_module()
class BHRL(TwoStageDetector):
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None,
                 training_type=None):
        super(BHRL, self).__init__(backbone=backbone,
                                                    neck=neck,
                                                    rpn_head=rpn_head,
                                                    roi_head=roi_head,
                                                    train_cfg=train_cfg,
                                                    test_cfg=test_cfg,
                                                    pretrained=pretrained,
                                                    init_cfg=init_cfg)

        self.matching_block = MatchModule(512, 384)
        if training_type == "Position":
            print("Training Type Position")
            self.stddev = nn.ParameterList([
                nn.Parameter(torch.full((1, 1, 48, 48), 0.1)),
                nn.Parameter(torch.full((1, 1, 24, 24), 0.1)),
                nn.Parameter(torch.full((1, 1, 12, 12), 0.1)),
                nn.Parameter(torch.full((1, 1, 6, 6), 0.1)),
                nn.Parameter(torch.full((1, 1, 3, 3), 0.1))
            ])
        if training_type == "Channel":
            print("Training Type Channel")
            self.stddev = nn.ParameterList([
                nn.Parameter(torch.full((1, 256, 1, 1), 0.1)),
                nn.Parameter(torch.full((1, 256, 1, 1), 0.1)),
                nn.Parameter(torch.full((1, 256, 1, 1), 0.1)),
                nn.Parameter(torch.full((1, 256, 1, 1), 0.1)),
                nn.Parameter(torch.full((1, 256, 1, 1), 0.1))
            ])        
        if training_type == "Single":
            print("Training Type Single")
            self.stddev = nn.ParameterList([
                nn.Parameter(torch.full((1, 1, 1, 1), 0.1)),
                nn.Parameter(torch.full((1, 1, 1, 1), 0.1)),
                nn.Parameter(torch.full((1, 1, 1, 1), 0.1)),
                nn.Parameter(torch.full((1, 1, 1, 1), 0.1)),
                nn.Parameter(torch.full((1, 1, 1, 1), 0.1))
            ]) 
        if training_type == "Position-Channel":
            print("Training Type Position-Channel")
            self.stddev = nn.ParameterList([
                nn.Parameter(torch.full((1, 256, 48, 48), 0.1)),
                nn.Parameter(torch.full((1, 256, 24, 24), 0.1)),
                nn.Parameter(torch.full((1, 256, 12, 12), 0.1)),
                nn.Parameter(torch.full((1, 256, 6, 6), 0.1)),
                nn.Parameter(torch.full((1, 256, 3, 3), 0.1))
            ])          
        if training_type == "Fixed":
            print("Training Type Fixed")
            self.stddev = [
                torch.full((1, 1, 1, 1), 0.1),
                torch.full((1, 1, 1, 1), 0.1),
                torch.full((1, 1, 1, 1), 0.1),
                torch.full((1, 1, 1, 1), 0.1),
                torch.full((1, 1, 1, 1), 0.1)
            ]

    def matching(self, img_feat, rf_feat):
        out = []
        for i in range(len(rf_feat)):
            out.append(self.matching_block(img_feat[i], rf_feat[i]))
        return out

    def extract_feat(self, img):
        img_feat = img[0]
        rf_feat = img[1]
        rf_bbox = img[2]
        img_feat = self.backbone(img_feat)
        rf_feat = self.backbone(rf_feat)
        if self.with_neck:
            img_feat = self.neck(img_feat)
            rf_feat = self.neck(rf_feat)
        img_feat_metric = self.matching(img_feat, rf_feat)
        ref_roi_feats = generate_ref_roi_feats(rf_feat, rf_bbox)
        return tuple(img_feat_metric), tuple(img_feat), ref_roi_feats

    def extract_feat_fork1(self, img):
        img_feat = img[0]
        rf_feat = img[1]
        rf_bbox = img[2]
        img_feat = self.backbone(img_feat)
        rf_feat = self.backbone(rf_feat)

        if self.with_neck:
            img_feat = self.neck(img_feat)
            rf_feat =  self.neck(rf_feat)
        rf_feat = add_gaussian_noise(rf_feat,std=self.stddev) # add noise to feature map of referred image
        img_feat_metric = self.matching(img_feat, rf_feat)
        ref_roi_feats = generate_ref_roi_feats(rf_feat, rf_bbox)
        return tuple(img_feat_metric), tuple(img_feat), ref_roi_feats

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        x, img_feat, ref_roi_feats = self.extract_feat_fork1(img)
        losses = dict()


        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
            
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_feat, ref_roi_feats, 
                                                 img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)
        
        return losses
        
    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        x, img_feat, ref_roi_feats = self.extract_feat(img)


        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas) 
        else:
            proposal_list = proposals
        return self.roi_head.simple_test(x, img_feat, ref_roi_feats, proposal_list, img_metas, rescale=rescale)