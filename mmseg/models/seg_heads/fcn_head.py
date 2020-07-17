import os
import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import auto_fp16, force_fp32, mask_target
from mmcv.cnn import ConvModule, constant_init, normal_init, build_upsample_layer
from mmdet.ops.carafe import CARAFEPack

from ..builder import HEADS
from ..builder import build_loss

@HEADS.register_module
class FCNHead(nn.Module):
    def __init__(self,
                 num_convs=2,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 drop_out=None,
                 upsample_first=False,
                 num_classes=81,
                 upsample_size=None,
                 upsample_method=None,
                 upsample_ratio=None,
                 upsample_cfg=dict(type='bilinear', scale_factor=2, upsample_size=None),
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_mask=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 loss_mask_aux=None,
                 loss_edge=None):
        super(FCNHead, self).__init__()
        self.upsample_cfg = upsample_cfg.copy()
        if self.upsample_cfg['type'] not in [
            None, 'deconv', 'nearest', 'bilinear', 'carafe'
        ]:
            raise ValueError(
                'Invalid upsample method {}, accepted methods '
                'are "deconv", "nearest", "bilinear", "carafe"'.format(
                    self.upsample_cfg['type']))
        if (loss_mask is None) and (loss_edge is None):
            raise ValueError(
                'Loss configuration must be provided.')
        if drop_out is not None:
            if not isinstance(drop_out, (list, tuple)):
                drop_out = [drop_out]
            assert len(drop_out) == num_convs
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_first = upsample_first
        self.upsample_method = self.upsample_cfg.get('type')
        self.scale_factor = self.upsample_cfg.pop('scale_factor', None)
        self.upsample_size = self.upsample_cfg.pop('upsample_size', None)
        self.num_classes = num_classes
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.loss_mask = build_loss(loss_mask) if loss_mask is not None else None
        self.loss_mask_aux = build_loss(loss_mask_aux) if loss_mask_aux is not None else None
        self.loss_edge = build_loss(loss_edge) if loss_edge is not None else None


        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
            if drop_out is not None:
                self.convs.append(nn.Dropout(drop_out[i]))
        upsample_in_channels = (
            self.conv_out_channels if self.num_convs > 0 else in_channels)

        upsample_cfg_ = self.upsample_cfg.copy()
        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            assert upsample_first
            upsample_cfg_.update(
                in_channels=upsample_in_channels,
                out_channels=self.conv_out_channels,
                kernel_size=self.scale_factor,
                stride=self.scale_factor)
        elif self.upsample_method == 'carafe':
            assert upsample_first
            upsample_cfg_.update(
                channels=upsample_in_channels, scale_factor=self.scale_factor)
        else:
            # suppress warnings
            align_corners = (None
                             if self.upsample_method == 'nearest' else False)
            if self.upsample_size:
                upsample_cfg_.update(
                    size=self.upsample_size,
                    mode=self.upsample_method,
                    align_corners=align_corners)
            else:
                upsample_cfg_.update(
                    scale_factor=self.scale_factor,
                    mode=self.upsample_method,
                    align_corners=align_corners)
        self.upsample = build_upsample_layer(upsample_cfg_)

        logits_in_channel = (
            self.conv_out_channels
            if self.upsample_method == 'deconv' else upsample_in_channels)
        self.conv_logits = nn.Conv2d(logits_in_channel, self.num_classes, 1)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        for m in [self.upsample, self.conv_logits]:
            if m is None:
                continue
            elif isinstance(m, nn.Upsample):
                continue
            elif isinstance(m, CARAFEPack):
                m.init_weights()
            else:
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    @auto_fp16()
    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        if self.upsample is not None:
            if self.upsample_first:
                x = self.upsample(x)
                if self.upsample_method == 'deconv':
                    x = self.relu(x)
                mask_pred = self.conv_logits(x)
            else:
                x = self.conv_logits(x)
                mask_pred = self.upsample(x)
        else:
            mask_pred = self.conv_logits(x)
        return mask_pred

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, mask_targets, valid_pixels=None):
        loss = dict()
        if self.loss_mask is not None:
            loss_mask = self.loss_mask(mask_pred, mask_targets, valid_pixels)
            loss['loss_mask'] = loss_mask
        if self.loss_mask_aux is not None:
            loss_mask_aux = self.loss_mask_aux(mask_pred, mask_targets, valid_pixels)
            loss['loss_mask_aux'] = loss_mask_aux
        if self.loss_edge is not None:
            loss_edge = self.loss_edge(mask_pred, mask_targets, valid_pixels)
            loss['loss_edge'] = loss_edge
        return loss


