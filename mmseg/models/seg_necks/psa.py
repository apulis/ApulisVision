"""Point-wise Spatial Attention Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import xavier_init
import math

from mmdet.core import auto_fp16
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer
from ..builder import NECKS


class _PointwiseSpatialAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 reduced_channels=512,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(_PointwiseSpatialAttention, self).__init__()

        self.collect_attention = _AttentionGeneration(in_channels, reduced_channels, out_channels, conv_cfg, norm_cfg)
        self.distribute_attention = _AttentionGeneration(in_channels, reduced_channels, out_channels, conv_cfg, norm_cfg)

    def forward(self, x):
        collect_fm = self.collect_attention(x)
        distribute_fm = self.distribute_attention(x)
        psa_fm = torch.cat([collect_fm, distribute_fm], dim=1)
        return psa_fm


class _AttentionGeneration(nn.Module):
    def __init__(self,
                 in_channels,
                 reduced_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(_AttentionGeneration, self).__init__()
        self.conv_reduce = ConvModule(
            in_channels,
            reduced_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.attention = nn.Sequential(
            ConvModule(
                reduced_channels,
                reduced_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg),
            build_conv_layer(
                conv_cfg,
                reduced_channels,
                out_channels,
                1,
                bias=False))
        self.reduced_channels = reduced_channels

    def forward(self, x):
        reduce_x = self.conv_reduce(x)
        attention = self.attention(reduce_x)
        n, c, h, w = attention.size()
        attention = attention.view(n, c, -1)
        reduce_x = reduce_x.view(n, self.reduced_channels, -1)
        fm = torch.bmm(reduce_x, torch.softmax(attention, dim=1))
        fm = fm.view(n, self.reduced_channels, h, w)
        return fm


@NECKS.register_module
class PSA(nn.Module):
    def __init__(self,
                 in_channels,
                 inter_channels=3600,
                 reduced_channels=512,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(PSA, self).__init__()
        # psa_out_channels = crop_size // 8 ** 2
        self.psa = _PointwiseSpatialAttention(in_channels, inter_channels, reduced_channels, conv_cfg, norm_cfg)
        self.conv_post = ConvModule(
            reduced_channels * 2,
            in_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, x):
        global_feature = self.psa(x)
        out = self.conv_post(global_feature)
        out = torch.cat([x, out], dim=1)
        return out

