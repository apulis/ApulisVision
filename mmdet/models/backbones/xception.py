import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

import torch.utils.checkpoint as cp
from .utils import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm
from ..builder import BACKBONES
from mmdet.models.plugins import GeneralizedAttention
from mmdet.ops import ContextBlock, DeformConv, ModulatedDeformConv
from mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
                      kaiming_init)


class SeparableConv2d(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 bias=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 gcb=None,
                 gen_attention=None):
        super(SeparableConv2d, self).__init__()
        assert dcn is None, "Not implemented yet."
        assert gen_attention is None, "Not implemented yet."
        assert gcb is None, "Not implemented yet."

        self.kernel_size = kernel_size
        self.dilation = dilation

        self.norm_name, norm = build_norm_layer(norm_cfg, inplanes)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            inplanes,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=inplanes,
            bias=bias)
        self.add_module(self.norm_name, norm)
        self.pointwise = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            1,
            bias=bias)

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def forward(self, x):
        x = self.fix_padding(x, self.kernel_size, self.dilation)
        x = self.conv1(x)
        x = self.norm(x)
        x = self.pointwise(x)

        return x

    def fix_padding(self, x, kernel_size, dilation):
        kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        padded_inputs = F.pad(x, (pad_beg, pad_end, pad_beg, pad_end))
        return padded_inputs


class Block(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 reps,
                 stride=1,
                 dilation=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 start_with_relu=True,
                 grow_first=True,
                 is_last=False,
                 dcn=None,
                 gcb=None,
                 gen_attention=None):
        super(Block, self).__init__()
        assert dcn is None, "Not implemented yet."
        assert gen_attention is None, "Not implemented yet."
        assert gcb is None, "Not implemented yet."

        self.skipnorm_name, skipnorm = build_norm_layer(norm_cfg, planes, postfix='skip')
        if planes != inplanes or stride != 1:
            self.skip = build_conv_layer(
                conv_cfg,
                inplanes,
                planes,
                1,
                stride=stride,
                bias=False
            )
            self.add_module(self.skipnorm_name, skipnorm)
        else:
            self.skip = None
        self.relu = nn.ReLU(True)
        rep = list()
        filters = inplanes
        if grow_first:
            if start_with_relu:
                rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation, conv_cfg=conv_cfg, norm_cfg=norm_cfg))
            rep.append(build_norm_layer(norm_cfg, planes)[1])
            filters = planes
        for i in range(reps - 1):
            if grow_first or start_with_relu:
                rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, 1, dilation, conv_cfg=conv_cfg, norm_cfg=norm_cfg))
            rep.append(build_norm_layer(norm_cfg, filters)[1])
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation, conv_cfg=conv_cfg, norm_cfg=norm_cfg))
        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 3, stride, conv_cfg=conv_cfg, norm_cfg=norm_cfg))
            rep.append(build_norm_layer(norm_cfg, planes)[1])
        elif is_last:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 3, 1, dilation, conv_cfg=conv_cfg, norm_cfg=norm_cfg))
            rep.append(build_norm_layer(norm_cfg, planes)[1])
        self.rep = nn.Sequential(*rep)

    @property
    def skipnorm(self):
        return getattr(self, self.skipnorm_name)

    def forward(self, x):
        out = self.rep(x)
        if self.skip is not None:
            skip = self.skipnorm(self.skip(x))
        else:
            skip = x
        out = out + skip
        return out

@BACKBONES.register_module
class Xception(nn.Module):
    """Modified Aligned Xception
    """

    def __init__(self,
                 depth,
                 in_channels=3,
                 output_stride=32,
                 num_stages=3,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 norm_eval=True):
        super(Xception, self).__init__()
        if depth not in [65, 71]:
            raise KeyError('invalid depth {} for xception'.format(depth))
        if output_stride == 32:
            entry_block3_stride = 2
            exit_block20_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 1)
        elif output_stride == 16:
            entry_block3_stride = 2
            exit_block20_stride = 1
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
        elif output_stride == 8:
            entry_block3_stride = 1
            exit_block20_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
        else:
            raise NotImplementedError

        self.depth = depth
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 3
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval

        # Entry flow
        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            32,
            3,
            stride=2,
            padding=1,
            bias=False)
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, 32, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(True)

        self.conv2 = build_conv_layer(
            conv_cfg,
            32,
            64,
            3,
            stride=1,
            padding=1,
            bias=False)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, 64, postfix=2)
        self.add_module(self.norm2_name, norm2)

        self.block1 = Block(64, 128, reps=2, stride=2, conv_cfg=conv_cfg, norm_cfg=norm_cfg, start_with_relu=False)
        if depth == 65:
            self.block2 = Block(128, 256, reps=2, stride=2, conv_cfg=conv_cfg, norm_cfg=norm_cfg, start_with_relu=False,
                                grow_first=True)
            self.block3 = Block(256, 728, reps=2, stride=entry_block3_stride, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                                start_with_relu=True, grow_first=True, is_last=True)
        elif depth == 71:
            self.block2 = nn.Sequential(
                Block(128, 256, reps=2, stride=2, conv_cfg=conv_cfg, norm_cfg=norm_cfg, start_with_relu=False, grow_first=True),
                Block(256, 728, reps=2, stride=2, conv_cfg=conv_cfg, norm_cfg=norm_cfg, start_with_relu=False, grow_first=True))
            self.block3 = Block(728, 728, reps=2, stride=entry_block3_stride, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                                start_with_relu=True, grow_first=True, is_last=True)

        # Middle flow
        midflow = list()
        for i in range(4, 20):
            midflow.append(Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, conv_cfg=conv_cfg,
                                 norm_cfg=norm_cfg, start_with_relu=True, grow_first=True))
        self.midflow = nn.Sequential(*midflow)

        # Exit flow
        self.block20 = Block(728, 1024, reps=2, stride=exit_block20_stride, dilation=exit_block_dilations[0],
                             conv_cfg=conv_cfg, norm_cfg=norm_cfg, start_with_relu=True, grow_first=False, is_last=True)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, dilation=exit_block_dilations[1], conv_cfg=conv_cfg, norm_cfg=norm_cfg)
        self.norm3_name, norm3 = build_norm_layer(norm_cfg, 1536, postfix=3)
        self.add_module(self.norm3_name, norm3)
        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=exit_block_dilations[1], conv_cfg=conv_cfg, norm_cfg=norm_cfg)
        self.norm4_name, norm4 = build_norm_layer(norm_cfg, 1536, postfix=4)
        self.add_module(self.norm4_name, norm4)
        self.conv5 = SeparableConv2d(1536, 2048, 3, 1, dilation=exit_block_dilations[1], conv_cfg=conv_cfg, norm_cfg=norm_cfg)
        self.norm5_name, norm5 = build_norm_layer(norm_cfg, 2048, postfix=5)
        self.add_module(self.norm5_name, norm5)

        self._freeze_stages()

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    @property
    def norm4(self):
        return getattr(self, self.norm4_name)

    @property
    def norm5(self):
        return getattr(self, self.norm5_name)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.norm1.eval()
            for m in [self.conv1, self.norm1]:
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

        else:
            raise TypeError('pretrained must be a str or None')


    def forward(self, x):
        outs = []
        # Entry flow
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.relu(x)
        # c1 = x
        outs.append(x)

        x = self.block2(x)
        # c2 = x
        x = self.block3(x)

        # Middle flow
        x = self.midflow(x)
        # c3 = x
        outs.append(x)

        # Exit flow
        x = self.block20(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.norm4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.norm5(x)
        x = self.relu(x)
        outs.append(x)

        return tuple(outs)

    def train(self, mode=True):
        super(Xception, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
