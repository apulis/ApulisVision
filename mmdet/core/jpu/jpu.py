##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Encoding Custermized NN Module"""
import os.path as osp
import sys

import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveAvgPool2d, BCELoss, CrossEntropyLoss
from torch.autograd import Variable

from mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
                      kaiming_init)



torch_ver = torch.__version__[:3]

__all__ = ['JPU']


class SeparableConv2d(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=False,
                 conv_cfg=None,
                 norm_cfg=None):
        super(SeparableConv2d, self).__init__()

        self.conv1 =  build_conv_layer(
            conv_cfg,
            inplanes,
            inplanes,
            kernel_size,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            groups=inplanes,
            bias=bias)
        self.bn = build_norm_layer(norm_cfg, inplanes)[1]
        self.pointwise = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class JPU(nn.Module):
    def __init__(self,
                 in_channels,
                 width=512,
                 mode='bilinear',
                 align_corners=True,
                 conv_cfg=None,
                 norm_cfg=None):
        super(JPU, self).__init__()
        self.mode = mode
        self.align_coners = align_corners

        self.conv5 = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                in_channels[-1],
                width,
                3,
                padding=1,
                bias=False),
            build_norm_layer(norm_cfg, width)[1],
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                in_channels[-2],
                width,
                3,
                padding=1,
                bias=False),
            build_norm_layer(norm_cfg, width)[1],
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                in_channels[-3],
                width,
                3,
                padding=1,
                bias=False),
            build_norm_layer(norm_cfg, width)[1],
            nn.ReLU(inplace=True))

        self.dilation1 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=1, dilation=1, bias=False),
                                       build_norm_layer(norm_cfg, width)[1],
                                       nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=2, dilation=2, bias=False),
                                       build_norm_layer(norm_cfg, width)[1],
                                       nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=4, dilation=4, bias=False),
                                       build_norm_layer(norm_cfg, width)[1],
                                       nn.ReLU(inplace=True))
        self.dilation4 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=8, dilation=8, bias=False),
                                       build_norm_layer(norm_cfg, width)[1],
                                       nn.ReLU(inplace=True))

    def forward(self, inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w), mode=self.mode, align_corners=self.align_coners)
        feats[-3] = F.interpolate(feats[-3], (h, w), mode=self.mode, align_corners=self.align_coners)
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)], dim=1)

        return inputs[0], inputs[1], inputs[2], feat

