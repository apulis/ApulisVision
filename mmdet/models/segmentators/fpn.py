"""Pyramid Scene Parsing Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .segbase import SegBaseModel
from ..builder import DETECTORS, build_backbone, build_head, build_neck



@DETECTORS.register_module
class FPN(SegBaseModel):

    def __init__(self,
                 backbone,
                 neck=None,
                 jpu=False,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(FPN, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        else:
            self.neck = None
        self.mask_head = build_head(mask_head)

        self.jpu = jpu
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)


    def init_weights(self, pretrained=None):
        super(FPN, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.mask_head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        features = self.backbone(img)
        x = self.neck(features)
        size = x[0].size()[2:]
        feature = x[0]
        for x_ in x[1:]:
            feature_ = F.interpolate(x_, size, mode='bilinear', align_corners=True)
            feature = torch.cat((feature, feature_), dim=1)
        return feature

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmedetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.mask_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_meta,
                      gt_semantic_seg,
                      valid_pixels=None):
        losses = dict()
        x = self.extract_feat(img)
        out = self.mask_head(x)
        loss = self.mask_head.loss(out, gt_semantic_seg, valid_pixels)
        losses.update(loss)

        return losses