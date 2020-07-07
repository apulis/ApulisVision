import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16
from mmcv.cnn import ConvModule, constant_init, normal_init, build_upsample_layer, build_norm_layer, build_conv_layer
from mmdet.ops.carafe import CARAFEPack
from ..builder import NECKS


class _ASPPConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 atrous_rate,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(_ASPPConv, self).__init__()
        self.norm_name, norm = build_norm_layer(norm_cfg, out_channels)
        self.block = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                in_channels,
                out_channels,
                3,
                padding=atrous_rate,
                dilation=atrous_rate,
                bias=False),
            norm,
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class _AsppPooling(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(_AsppPooling, self).__init__()
        self.norm_name, norm = build_norm_layer(norm_cfg, out_channels)
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            build_conv_layer(
                conv_cfg,
                in_channels,
                out_channels,
                1,
                bias=False),
            norm,
            nn.ReLU(True)
        )

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out


@NECKS.register_module
class ASPP(nn.Module):
    def __init__(self,
                 in_channels,
                 atrous_rates,
                 out_channels=256,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(ASPP, self).__init__()
        self.norm_name, norm = build_norm_layer(norm_cfg, out_channels)
        self.b0 = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                in_channels,
                out_channels,
                1,
                bias=False),
            norm,
            nn.ReLU(True)
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = _ASPPConv(in_channels, out_channels, rate1, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
        self.b2 = _ASPPConv(in_channels, out_channels, rate2, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
        self.b3 = _ASPPConv(in_channels, out_channels, rate3, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
        self.b4 = _AsppPooling(in_channels, out_channels, conv_cfg=conv_cfg, norm_cfg=norm_cfg)

        self.project = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                5 * out_channels,
                out_channels,
                1,
                bias=False),
            norm,
            nn.ReLU(True),
            nn.Dropout(0.5)
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, x):
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        feat5 = self.b4(x)
        x = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        x = self.project(x)
        return x


@NECKS.register_module
class ASPPPlus(nn.Module):
    def __init__(self,
                 in_channels,
                 atrous_rates,
                 out_channels=256,
                 out_channels_aux=48,
                 aux_level=0,
                 upsample_cfg=dict(type='bilinear', scale_factor=None, upsample_size=None),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(ASPPPlus, self).__init__()
        assert isinstance(aux_level, (int, list))
        self.aux_level = [aux_level] if isinstance(aux_level, int) else aux_level
        assert len(self.aux_level) <= 2
        self.upsample_cfg = upsample_cfg.copy()
        if self.upsample_cfg['type'] not in [
            None, 'deconv', 'nearest', 'bilinear', 'carafe'
        ]:
            raise ValueError(
                'Invalid upsample method {}, accepted methods '
                'are "deconv", "nearest", "bilinear", "carafe"'.format(
                    self.upsample_cfg['type']))
        self.upsample_method = self.upsample_cfg.get('type')
        self.upsample_size = self.upsample_cfg.pop('upsample_size')
        self.aspp = ASPP(in_channels[-1], atrous_rates, out_channels, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
        self.c1_block = ConvModule(in_channels[self.aux_level[0]], out_channels_aux, 3, padding=1, conv_cfg=conv_cfg,
                                   norm_cfg=norm_cfg, act_cfg=dict(type='ReLU'), inplace=False)

        upsample_cfg_ = self.upsample_cfg.copy()
        align_corners = (None
                         if self.upsample_method == 'nearest' else False)
        if self.upsample_method in ['nearest', 'bilinear']:
            if self.upsample_size:
                upsample_cfg_.update(
                    size=self.upsample_size,
                    mode=self.upsample_method,
                    align_corners=align_corners)
        else:
            assert isinstance(self.upsample_cfg.scale_factor, (int, list))
            self.scale_factors = [self.upsample_cfg.scale_factor] if isinstance(self.upsample_cfg.scale_factor,
                                                                                int) else self.upsample_cfg.scale_factor
            assert len(self.scale_factors) == len(self.aux_level)

        if self.upsample_method is None:
            self.upsample1 = None
        elif self.upsample_method == 'deconv':
            upsample_cfg_.update(
                in_channels=out_channels_aux,
                out_channels=out_channels_aux,
                kernel_size=self.scale_factors[0],
                stride=self.scale_factors[0])
        elif self.upsample_method == 'carafe':
            upsample_cfg_.update(
                channels=out_channels_aux, scale_factor=self.scale_factor[0])
        else:
            if not self.upsample_size:
                upsample_cfg_.update(
                    scale_factor=self.scale_factors[0],
                    mode=self.upsample_method,
                    align_corners=align_corners)
        self.upsample1 = build_upsample_layer(upsample_cfg_)

        if len(self.aux_level) > 1:
            self.c2_block = ConvModule(out_channels_aux+in_channels[self.aux_level[1]], out_channels_aux, 3, padding=1,
                                       conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=dict(type='ReLU'), inplace=False)
            if self.upsample_method is None:
                self.upsample2 = None
            elif self.upsample_method == 'deconv':
                upsample_cfg_.update(
                    in_channels=in_channels[self.aux_level[1]],
                    out_channels=in_channels[self.aux_level[1]],
                    kernel_size=self.scale_factors[1],
                    stride=self.scale_factors[1])
            elif self.upsample_method == 'carafe':
                upsample_cfg_.update(
                    channels=in_channels[self.aux_level[1]], scale_factor=self.scale_factor[1])
            else:
                if not self.upsample_size:
                    upsample_cfg_.update(
                        scale_factor=self.scale_factors[1],
                        mode=self.upsample_method,
                        align_corners=align_corners)
            self.upsample2 = build_upsample_layer(upsample_cfg_)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, x):
        size = x[self.aux_level[0]].size()[2:]
        c = self.c1_block(x[self.aux_level[0]])
        if len(self.aux_level) > 1:
            c_ = self.upsample2(x[self.aux_level[1]])
            c = self.c2_block(torch.cat([c, c_], dim=1))
        x = self.aspp(x[-1])
        x = self.upsample1(x)
        return torch.cat([x, c], dim=1)


'''
@NECKS.register_module
class DeepAggregation(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 atrous_rates,
                 upsample_method='bilinear',
                 upsample_ratio=[2, 4],
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(DeepAggregation, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_method=upsample_method
        if upsample_method == 'bilinear':
            self.upsample1 = nn.Upsample(
                scale_factor=upsample_ratio[0], mode=self.upsample_method, align_corners=True)
            self.upsample2 = nn.Upsample(
                scale_factor=upsample_ratio[1], mode=self.upsample_method, align_corners=True)
        else:
            self.upsample1 = nn.ConvTranspose2d(in_channels[1], in_channels[1], upsample_ratio[0], stride=2)
            self.upsample2 = nn.ConvTranspose2d(in_channels[1], in_channels[1], upsample_ratio[1], stride=2)
        self.aspp = ASPP(in_channels[-1], atrous_rates, out_channels, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
        self.c1_block = ConvModule(in_channels[0], in_channels[1], 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, activation='relu', inplace=False)
        self.c2_block = ConvModule(in_channels[1] * 2, out_channels, kernel_size=3, padding=1,
                                   conv_cfg=conv_cfg, norm_cfg=norm_cfg, activation='relu', inplace=False)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, x):
        #size = x[0].size()[2:]
        c1 = self.c1_block(x[0])
        c2 = self.upsample1(x[1])
        c = self.c2_block(torch.cat([c1, c2], dim=1))
        x = self.aspp(x[-1])
        x = self.upsample2(x)
        return torch.cat([x, c], dim=1)
'''