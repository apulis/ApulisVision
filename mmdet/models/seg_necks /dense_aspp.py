import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16
from mmcv.cnn import ConvModule, constant_init, normal_init, build_conv_layer, build_norm_layer, build_upsample_layer
from mmdet.ops.carafe import CARAFEPack
from ..builder import NECKS

class _DenseAsppBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 num_channels1,
                 num_channels2,
                 atrous_rate,
                 drop_out,
                 bn_start=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(_DenseAsppBlock, self).__init__()
        if bn_start:
            self.b1 = nn.Sequential(
                build_norm_layer(norm_cfg, in_channels)[1],
                nn.ReLU(True),
                build_conv_layer(
                    conv_cfg,
                    in_channels,
                    num_channels1,
                    1,
                    bias=False)
            )
        else:
            self.b1 = nn.Sequential(
                nn.ReLU(True),
                build_conv_layer(
                    conv_cfg,
                    in_channels,
                    num_channels1,
                    1,
                    bias=False)
            )

        self.b2 = nn.Sequential(
            build_norm_layer(norm_cfg, num_channels1)[1],
            nn.ReLU(True),
            build_conv_layer(
                conv_cfg,
                num_channels1,
                num_channels2,
                3,
                dilation=atrous_rate,
                padding=atrous_rate,
                bias=False),
                nn.Dropout(drop_out)
        )

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        return x



@NECKS.register_module
class DenseASPP(nn.Module):
    def __init__(self,
                 num_features,
                 d_feature0,
                 d_feature1,
                 atrous_rates,
                 drop_out,
                 reduction_channels=512,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(DenseASPP, self).__init__()
        if reduction_channels:
            self.b0 = nn.Sequential(
                build_conv_layer(
                    conv_cfg,
                    num_features,
                    reduction_channels,
                    1,
                    bias=False),
                build_norm_layer(norm_cfg, reduction_channels)[1]
            )
            num_features = reduction_channels
        else:
            self.b0 = nn.Identity()

        rate1, rate2, rate3, rate4, rate5 = tuple(atrous_rates)
        self.ASPP_3 = _DenseAsppBlock(in_channels=num_features, num_channels1=d_feature0, num_channels2=d_feature1,
                                      atrous_rate=rate1, drop_out=drop_out, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                                      bn_start=False)

        self.ASPP_6 = _DenseAsppBlock(in_channels=num_features + d_feature1 * 1, num_channels1=d_feature0, num_channels2=d_feature1,
                                      atrous_rate=rate2, drop_out=drop_out, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                                      bn_start=True)

        self.ASPP_12 = _DenseAsppBlock(in_channels=num_features + d_feature1 * 2, num_channels1=d_feature0, num_channels2=d_feature1,
                                       atrous_rate=rate3, drop_out=drop_out, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                                       bn_start=True)

        self.ASPP_18 = _DenseAsppBlock(in_channels=num_features + d_feature1 * 3, num_channels1=d_feature0, num_channels2=d_feature1,
                                       atrous_rate=rate4, drop_out=drop_out, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                                       bn_start=True)

        self.ASPP_24 = _DenseAsppBlock(in_channels=num_features + d_feature1 * 4, num_channels1=d_feature0, num_channels2=d_feature1,
                                       atrous_rate=rate4, drop_out=drop_out, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                                       bn_start=True)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')


    def forward(self, x):
        x = self.b0(x)

        aspp3 = self.ASPP_3(x)
        x = torch.cat((aspp3, x), dim=1)

        aspp6 = self.ASPP_6(x)
        x = torch.cat((aspp6, x), dim=1)

        aspp12 = self.ASPP_12(x)
        x = torch.cat((aspp12, x), dim=1)

        aspp18 = self.ASPP_18(x)
        x = torch.cat((aspp18, x), dim=1)

        aspp24 = self.ASPP_24(x)
        x = torch.cat((aspp24, x), dim=1)
        return x


@NECKS.register_module
class DenseASPPPlus(nn.Module):
    def __init__(self,
                 in_channels,
                 d_feature0,
                 d_feature1,
                 atrous_rates,
                 drop_out=0.1,
                 reduction_channels=512,
                 out_channels_aux=48,
                 aux_level=0,
                 upsample_cfg=dict(type='bilinear', scale_factor=None, upsample_size=None),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(DenseASPPPlus, self).__init__()
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
        self.aspp = DenseASPP(in_channels[-1], d_feature0, d_feature1, atrous_rates, drop_out, reduction_channels, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
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