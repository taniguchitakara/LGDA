import torch
import torch.nn as nn

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import DETECTORS, build_roi_extractor, build_head, HEADS
from .two_stage import TwoStageDetector
from ..plugins.match_module import MatchModule
from ..plugins.generate_ref_roi_feats import generate_ref_roi_feats, generate_ref_roi_feats_folk, add_gaussian_noise,add_gaussian_noise_old
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
                 init_cfg=None):
        super(BHRL, self).__init__(backbone=backbone,
                                                    neck=neck,
                                                    rpn_head=rpn_head,
                                                    roi_head=roi_head,
                                                    train_cfg=train_cfg,
                                                    test_cfg=test_cfg,
                                                    pretrained=pretrained,
                                                    init_cfg=init_cfg)

        self.matching_block = MatchModule(512, 384)
        '''
        self.stddev = nn.ParameterList([
            nn.Parameter(torch.full((1, 256, 48, 48), 0.001)),
            nn.Parameter(torch.full((1, 256, 24, 24), 0.001)),
            nn.Parameter(torch.full((1, 256, 12, 12), 0.001)),
            nn.Parameter(torch.full((1, 256, 6, 6), 0.001)),
            nn.Parameter(torch.full((1, 256, 3, 3), 0.001))
        ])
        '''
        # self.stddev = nn.Parameter(torch.tensor(0.001)) #old version



        self.stddev_for_test = torch.tensor(0.9)

    def matching(self, img_feat, rf_feat):
        out = []
        for i in range(len(rf_feat)):
            out.append(self.matching_block(img_feat[i], rf_feat[i]))
        return out

    def extract_feat(self, img):
        #print(img[0])
        img_feat = img[0]
        #print(img[2])
        rf_feat = img[1]
        rf_bbox = img[2]
        img_feat = self.backbone(img_feat)
        rf_feat = self.backbone(rf_feat)
        if self.with_neck:
            img_feat = self.neck(img_feat)
            rf_feat = self.neck(rf_feat)
            #rf_feat_2 = copy.deepcopy(add_gaussian_noise(rf_feat))
            #rf_feat = add_gaussian_noise(rf_feat)

        img_feat_metric = self.matching(img_feat, rf_feat)
        ##img_feat_metric_2 = self.matching(img_feat,rf_feat_2)
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
        rf_feat = add_gaussian_noise_old(rf_feat,std=self.stddev)#add noise to feature map of referred image
        img_feat_metric = self.matching(img_feat, rf_feat)
        #print(self.stddev_for_test)
        ref_roi_feats = generate_ref_roi_feats(rf_feat, rf_bbox)
        #print(self.stddev)
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
        #print(img)
        x, img_feat, ref_roi_feats = self.extract_feat_fork1(img)
        x_folk, img_feat, ref_roi_feats_folk = self.extract_feat_fork1(img)
        """
        x_folk2, img_feat, ref_roi_feats_folk2 = self.extract_feat_fork1(img)
        x_folk3, img_feat, ref_roi_feats_folk3 = self.extract_feat_fork1(img)
        """
        #x, img_feat, ref_roi_feats_folk2 = self.extract_feat_fork1(img,stddev=self.stddev)
        #x, img_feat, ref_roi_feats_folk3 = self.extract_feat_fork1(img,stddev=self.stddev)
        losses = dict()
        #losses_folk = dict()
        #rpn_total_losses = dict()
        #roi_total_losses = dict()


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
            
            rpn_losses_folk, proposal_list_folk = self.rpn_head.forward_train(
                x_folk,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)  
            """
            rpn_losses_folk2, proposal_list_folk2 = self.rpn_head.forward_train(
                x_folk2,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)  
            
            rpn_losses_folk3, proposal_list_folk3 = self.rpn_head.forward_train(
                x_folk3,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            """
            #print(rpn_losses)
            #print(rpn_losses_folk)
            for key in rpn_losses.keys():
                rpn_losses[key][0] = (rpn_losses_folk[key][0] + rpn_losses[key][0])/2 #+ roi_losses_folk2[key] + roi_losses_folk3[key])/4
            losses.update(rpn_losses)
            #print(rpn_losses)
        #else:
        #    proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_feat, ref_roi_feats, 
                                                 img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        roi_losses_folk = self.roi_head.forward_train(x_folk, img_feat, ref_roi_feats_folk, 
                                                 img_metas, proposal_list_folk,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        """
        roi_losses_folk2 = self.roi_head.forward_train(x_folk2, img_feat, ref_roi_feats_folk2, 
                                                 img_metas, proposal_list_folk2,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        roi_losses_folk3 = self.roi_head.forward_train(x_folk3, img_feat, ref_roi_feats_folk3, 
                                                 img_metas, proposal_list_folk3,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        """
        for key in roi_losses.keys():
            roi_losses[key]= (roi_losses_folk[key] + roi_losses[key])/2 #+ roi_losses_folk2[key] + roi_losses_folk3[key])/4
            #print(key)
        #self.optimizer.step()
        #total_loss['s0.loss_bbox'].backward(retain_graph=True)
        losses.update(roi_losses)
        #print(self.stddev)
        #print(losses)
        #print(self.stddev)
        return losses
        
    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        x, img_feat, ref_roi_feats = self.extract_feat(img)
        #x_folk, _, ref_roi_feats_folk = self.extract_feat_fork1(img)
        #x_folk2, img_feat, ref_roi_feats_folk2 = self.extract_feat_fork1(img)
        #x_folk3, img_feat, ref_roi_feats_folk3 = self.extract_feat_fork1(img)
        #x, img_feat, ref_roi_feats_folk = self.extract_feat_fork1(img,stddev=self.stddev)
        #x, img_feat, ref_roi_feats_folk2 = self.extract_feat_fork1(img,stddev=self.stddev)


        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas) 
            #proposal_list_folk = self.rpn_head.simple_test_rpn(x_folk, img_metas)
            #proposal_list_folk2 = self.rpn_head.simple_test_rpn(x_folk2, img_metas)
            #proposal_list_folk3 = self.rpn_head.simple_test_rpn(x_folk3, img_metas)
        else:
            proposal_list = proposals
            #proposal_list_folk = proposals
        #print(self.roi_head.simple_test(x, img_feat, ref_roi_feats, proposal_list, img_metas, rescale=rescale)[0])
        """
        ans_list = [[]]
        ans = np.concatenate((np.array(self.roi_head.simple_test(x, img_feat, ref_roi_feats, proposal_list, img_metas, rescale=rescale)[0][0])
                         ,np.array(self.roi_head.simple_test(x_folk, img_feat, ref_roi_feats_folk, proposal_list_folk, img_metas, rescale=rescale)[0][0]),
                         np.array(self.roi_head.simple_test(x_folk2, img_feat, ref_roi_feats_folk2, proposal_list_folk2, img_metas, rescale=rescale)[0][0])))
        sorted_ans = ans[ans[:,4].argsort()[::-1]]
        
        #print(sorted_ans)
        ans_list[0].append(sorted_ans)
        
        return ans_list
        """
        #print(1)
        #return self.roi_head.simple_test_folk(img_feat, ref_roi_feats, ref_roi_feats_folk, proposal_list, proposal_list_folk, img_metas, rescale=rescale)
        #print(self.roi_head.simple_test(x, img_feat, ref_roi_feats, proposal_list, img_metas, rescale=rescale))
        return self.roi_head.simple_test(x, img_feat, ref_roi_feats, proposal_list, img_metas, rescale=rescale)