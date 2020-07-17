"""Dual Attention Network"""
import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import auto_fp16, force_fp32, mask_target
from mmcv.cnn import ConvModule, constant_init, normal_init, build_conv_layer, build_upsample_layer
from mmdet.ops.carafe import CARAFEPack

from ..builder import HEADS
from ..builder import build_loss

class _PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self,
                 in_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = build_conv_layer(
            conv_cfg,
            in_channels,
            in_channels // 8,
            1)
        self.conv_c = build_conv_layer(
            conv_cfg,
            in_channels,
            in_channels // 8,
            1)
        self.conv_d = build_conv_layer(
            conv_cfg,
            in_channels,
            in_channels,
            1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


class _ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self):
        super(_ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out


@HEADS.register_module
class DAHead(nn.Module):
    def __init__(self,
                 in_channel,
                 drop_out=0.1,
                 num_classes=81,
                 upsample_cfg=dict(type='bilinear', scale_factor=2, upsamle_size=None),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 loss_mask=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 loss_mask_aux=None,
                 loss_mask_aux_layer=None,
                 loss_edge=None):
        super(DAHead, self).__init__()
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
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.upsample_method = self.upsample_cfg.get('type')
        self.scale_factor = self.upsample_cfg.pop('scale_factor', None)
        self.upsample_size = self.upsample_cfg.pop('upsample_size', None)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False

        self.loss_mask = build_loss(loss_mask) if loss_mask is not None else None
        self.loss_mask_aux = build_loss(loss_mask_aux) if loss_mask_aux is not None else None
        self.loss_mask_aux_layer = build_loss(loss_mask_aux_layer) if loss_mask_aux_layer is not None else None
        self.loss_edge = build_loss(loss_edge) if loss_edge is not None else None


        inter_channel = in_channel // 4
        self.conv_p1 = ConvModule(
                in_channel,
                inter_channel,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                inplace=True)
        self.conv_c1 = ConvModule(
                in_channel,
                inter_channel,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                inplace=True)
        self.pam = _PositionAttentionModule(inter_channel, conv_cfg, norm_cfg)
        self.cam = _ChannelAttentionModule()
        self.conv_p2 = ConvModule(
                inter_channel,
                inter_channel,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                inplace=True)
        self.conv_c2 = ConvModule(
                inter_channel,
                inter_channel,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                inplace=True)

        self.conv_logits = nn.Sequential(
            nn.Dropout(drop_out),
            build_conv_layer(
                conv_cfg,
                inter_channel,
                num_classes,
                1,
                bias=True))
        self.conv_p3 = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                inter_channel,
                num_classes,
                1,
                bias=True))
        self.conv_c3 = nn.Sequential(
            nn.Dropout(drop_out),
            build_conv_layer(
                conv_cfg,
                inter_channel,
                num_classes,
                1,
                bias=True))

        upsample_cfg_ = self.upsample_cfg.copy()
        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            upsample_cfg_.update(
                in_channels=num_classes,
                out_channels=num_classes,
                kernel_size=self.scale_factor,
                stride=self.scale_factor)
        elif self.upsample_method == 'carafe':
            upsample_cfg_.update(
                channels=num_classes, scale_factor=self.scale_factor)
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

    def init_weights(self):
        #for m in [self.upsample, self.conv_logits]:
        for m in [self.conv_logits]:
            if m is None:
                continue
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    @auto_fp16()
    def forward(self, x):
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)

        feat_c = self.conv_c1(x)
        feat_c = self.cam(feat_c)
        feat_c = self.conv_c2(feat_c)

        feat_fusion = feat_p + feat_c

        outputs = []
        fusion_out = self.conv_logits(feat_fusion)
        p_out = self.conv_p3(feat_p)
        c_out = self.conv_c3(feat_c)
        if self.upsample is not None:
            outputs.append(self.upsample(p_out))
            outputs.append(self.upsample(c_out))
            outputs.append(self.upsample(fusion_out))
        else:
            outputs.append(p_out)
            outputs.append(c_out)
            outputs.append(fusion_out)
        return tuple(outputs)

    @force_fp32(apply_to=('mask_pred',))
    def loss(self, mask_pred, mask_targets, valid_pixels=None):
        loss = dict()
        if self.loss_mask is not None:
            loss_mask = self.loss_mask(mask_pred[-1], mask_targets, valid_pixels)
            loss['loss_mask'] = loss_mask
        if self.loss_mask_aux is not None:
            loss_mask_aux = self.loss_mask_aux(mask_pred[-1], mask_targets, valid_pixels)
            loss['loss_mask_aux'] = loss_mask_aux
        if self.loss_mask_aux_layer is not None:
            loss_mask_aux_layer = (self.loss_mask_aux_layer(mask_pred[0], mask_targets,
                                                            valid_pixels) + self.loss_mask_aux_layer(mask_pred[1],
                                                                                                     mask_targets,
                                                                                                     valid_pixels)) / 2
            loss['loss_mask_aux_layer'] = loss_mask_aux_layer
        if self.loss_edge is not None:
            loss_edge = self.loss_edge(mask_pred[-1], mask_targets, valid_pixels)
            loss['loss_edge'] = loss_edge
        return loss
