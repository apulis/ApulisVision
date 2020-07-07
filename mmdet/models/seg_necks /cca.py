"""Criss-Cross Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.autograd.function import once_differentiable
from mmcv.cnn import xavier_init
import math

from core.nn import _C
from mmdet.core import auto_fp16
from mmcv.cnn import ConvModule, constant_init, normal_init, build_conv_layer, build_norm_layer
from ..builder import NECKS

class _CAWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, f):
        weight = _C.ca_forward(t, f)

        ctx.save_for_backward(t, f)

        return weight

    @staticmethod
    @once_differentiable
    def backward(ctx, dw):
        t, f = ctx.saved_tensors

        dt, df = _C.ca_backward(dw, t, f)
        return dt, df


class _CAMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, g):
        out = _C.ca_map_forward(weight, g)

        ctx.save_for_backward(weight, g)

        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        weight, g = ctx.saved_tensors

        dw, dg = _C.ca_map_backward(dout, weight, g)

        return dw, dg


ca_weight = _CAWeight.apply
ca_map = _CAMap.apply


class CrissCrossAttention(nn.Module):
    """Criss-Cross Attention Module"""

    def __init__(self,
                 in_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = build_conv_layer(
            conv_cfg,
            in_channels,
            in_channels // 8,
            1)
        self.key_conv = build_conv_layer(
            conv_cfg,
            in_channels,
            in_channels // 8,
            1)
        self.value_conv = build_conv_layer(
            conv_cfg,
            in_channels,
            in_channels,
            1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)

        energy = ca_weight(proj_query, proj_key)
        attention = F.softmax(energy, 1)
        out = ca_map(attention, proj_value)
        out = self.gamma * out + x

        return out


@NECKS.register_module
class RCCA(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=512,
                 drop_out=0.1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(RCCA, self).__init__()
        inter_channels = in_channels // 4
        self.conva = ConvModule(
            in_channels,
            inter_channels,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.cca = CrissCrossAttention(inter_channels, conv_cfg, norm_cfg)
        self.convb = ConvModule(
            inter_channels,
            inter_channels,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.bottleneck = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                in_channels + inter_channels,
                out_channels,
                3,
                padding=1,
                bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            nn.Dropout2d(0.1))


    def init_weights(self):
        for m in self.modules():
            #if isinstance(m, nn.Conv2d):
                #xavier_init(m, distribution='uniform')
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

    def forward(self, x, recurrence=1):
        out = self.conva(x)
        for i in range(recurrence):
            out = self.cca(out)
        out = self.convb(out)
        out = torch.cat([x, out], dim=1)
        out = self.bottleneck(out)
        return out