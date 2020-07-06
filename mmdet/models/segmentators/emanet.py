"""Pyramid Scene Parsing Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .segbase import SegBaseModel
from .fcn import FCN
from objdet.models.builder import DETECTORS
from objdet.models import builder


@DETECTORS.register_module
class EMANet(FCN):
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
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(EMANet, self).__init__(backbone=backbone,
                                     neck=neck,
                                     jpu=jpu,
                                     mask_head=mask_head,
                                     aux_head=aux_head,
                                     train_cfg=train_cfg,
                                     test_cfg=test_cfg,
                                     pretrained=pretrained)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        features = self.backbone(img)
        x, mu = self.neck(features[-1])
        return x, mu

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmedetection/tools/get_flops.py`
        """
        x, mu = self.extract_feat(img)
        outs = self.mask_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_meta,
                      gt_semantic_seg,
                      valid_pixels=None):
        losses = dict()
        features = self.backbone(img)
        x, mu = self.neck(features[-1])
        out = self.mask_head(x)
        loss = self.mask_head.loss(out, gt_semantic_seg, valid_pixels)
        losses.update(loss)
        if self.aux_head is not None:
            auxout = self.aux_head(features[-2])
            loss_aux = self.aux_head.loss(auxout, gt_semantic_seg, valid_pixels)
            assert len(loss_aux) == 1
            losses['loss_aux'] = loss_aux['loss_mask']

        losses_ = dict()
        losses_['mu'] = mu
        losses_['losses'] = losses
        return losses_

    def simple_test(self, img, img_meta, valid_pixels=None, **kwargs):
        x = self.extract_feat(img)
        preds = self.mask_head(x).cpu().numpy()
        seg_results = np.argmax(preds, axis=1)
        if valid_pixels is not None:
            valid_pixels = valid_pixels.cpu().numpy().astype(np.uint8)
            seg_results = seg_results * valid_pixels
        return seg_results