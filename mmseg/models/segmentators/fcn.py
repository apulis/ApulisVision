"""Pyramid Scene Parsing Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .segbase import SegBaseModel
from ..builder import DETECTORS, build_backbone, build_head, build_neck


@DETECTORS.register_module
class FCN(SegBaseModel):
    r"""FCN

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.

    Reference:
        Chen, Liang-Chieh, et al. "Rethinking atrous convolution for semantic image segmentation."
        arXiv preprint arXiv:1706.05587 (2017).
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 jpu=False,
                 mask_head=None,
                 aux_head=None,
                 aux_level=-2,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(FCN, self).__init__()
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck) if neck is not None else None
        self.mask_head = build_head(mask_head)
        if aux_head is not None:
            self.aux_head = build_head(aux_head)
            self.aux_level = aux_level
        else:
            self.aux_head = None

        self.jpu = jpu
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)


    def init_weights(self, pretrained=None):
        super(FCN, self).init_weights(pretrained)
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
        x = self.neck(features[-1])
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmedetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.mask_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg, 
                      valid_pixels=None,
                      **kwargs):
        losses = dict()
        features = self.backbone(img)
        if self.with_neck:
            features_ = self.neck(features[-1])
        else:
            features_ = features[-1]
        out = self.mask_head(features_)
        loss = self.mask_head.loss(out, gt_semantic_seg, valid_pixels)
        losses.update(loss)
        if self.aux_head is not None:
            auxout = self.aux_head(features[self.aux_level])
            loss_aux = self.aux_head.loss(auxout, gt_semantic_seg, valid_pixels)
            assert len(loss_aux) == 1
            losses['loss_aux'] = loss_aux['loss_mask']

        return losses

    def simple_test(self, img, img_metas=None, valid_pixels=None, **kwargs):
        x = self.extract_feat(img)
        preds = self.mask_head(x).cpu().numpy()
        seg_results = np.argmax(preds, axis=1)
        if valid_pixels is not None:
            valid_pixels = valid_pixels.cpu().numpy().astype(np.uint8)
            seg_results = seg_results * valid_pixels
        return seg_results