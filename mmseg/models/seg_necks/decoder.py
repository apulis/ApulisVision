import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16
from mmcv.cnn import ConvModule, constant_init, normal_init
from ..builder import NECKS

class sSE(nn.Module):
    def __init__(self, out_channels, conv_cfg=None, norm_cfg=None):
        super(sSE, self).__init__()
        self.conv = ConvModule(
                        out_channels,
                        1,
                        kernel_size=1,
                        padding=0,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=None)

    def forward(self, x):
        x = self.conv(x)
        # print('spatial',x.size())
        x = F.sigmoid(x)
        return x


class cSE(nn.Module):
    def __init__(self, out_channels, conv_cfg=None, norm_cfg=None):
        super(cSE, self).__init__()
        self.conv1 = ConvModule(
            out_channels,
            out_channels // 2,
            kernel_size=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.conv2 = ConvModule(
            out_channels // 2,
            out_channels,
            kernel_size=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

    def forward(self, x):
        x = nn.AvgPool2d(x.size()[2:])(x)
        # print('channel',x.size())
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.sigmoid(x)
        return x


class Up(nn.Module):
    def __init__(self,
                 in_ch1,
                 in_ch2,
                 out_ch,
                 type='unet',
                 upsample_method='bilinear',
                 conv_cfg=None,
                 norm_cfg=None):
        super(Up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        self.type = type
        if upsample_method == 'bilinear':
            self.upsample = nn.Upsample(
                scale_factor=2, mode=upsample_method, align_corners=True)
        else:
            self.upsample = nn.ConvTranspose2d(in_ch1, in_ch1, 2, stride=2)

        if type == 'unet':
            self.conv = nn.Sequential(
                ConvModule(
                    in_ch1 + in_ch2,
                    out_ch,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg),
                ConvModule(
                    out_ch,
                    out_ch,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
        elif type == 'se':
            # Squeeze & Excitation
            inter_ch = in_ch2 if in_ch2 > 0 else in_ch1 // 2
            self.conv = nn.Sequential(
                ConvModule(
                    in_ch1 + in_ch2,
                    inter_ch,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg),
                ConvModule(
                    inter_ch,
                    out_ch,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg)
            )
            self.spatial_gate = sSE(out_ch)
            self.channel_gate = cSE(out_ch)
        else:
            raise NotImplementedError


    def forward(self, x1, x2=None):
        x1 = self.upsample(x1)

        if x2 is not None:
            # input is CHW
            #diffY = x2.size()[2] - x1.size()[2]
            #diffX = x2.size()[3] - x1.size()[3]

            #x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

            # for padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        x = self.conv(x)
        if self.type == 'se':
            g1 = self.spatial_gate(x)
            g2 = self.channel_gate(x)
            x = g1 * x + g2 * x
        return x


@NECKS.register_module
class Decoder(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channel,
                 inter_channels,
                 center_channels,
                 mode='unet',
                 upsample_method='bilinear',
                 out_type='last',
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 activation=None):
        super(Decoder, self).__init__()
        assert isinstance(in_channels, list)
        assert isinstance(inter_channels, list)
        assert out_type in ['last', 'all', 'concat']
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.center_channels = center_channels
        self.out_channel = out_channel
        self.out_type = out_type
        self.num_ins = len(in_channels)
        self.activation = activation
        self.fp16_enabled = False

        self.lateral_convs = nn.ModuleList()

        self.center = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvModule(
                in_channels[-1],
                in_channels[-1],
                kernel_size=3,
                padding=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg),
            ConvModule(
                in_channels[-1],
                center_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg))

        self.decoders = nn.ModuleList()
        in_ch1 = center_channels
        for i in range(self.num_ins):
            decoder = Up(in_ch1, in_channels[-(i+1)], inter_channels[i], mode, upsample_method, conv_cfg, norm_cfg)
            self.decoders.append(decoder)
            in_ch1 = inter_channels[i]
        self.decoders.append(
            Up(in_ch1, 0, out_channel, mode, upsample_method, conv_cfg, norm_cfg))

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        outs = []
        x = self.center(inputs[-1])
        for i, decoder in enumerate(self.decoders[:-1]):
            x = decoder(x, inputs[-(i+1)])
            outs.append(x)
        x = self.decoders[-1](x)
        outs.append(x)

        if self.out_type == 'concat':
            feature = outs[-1]
            for i, x_ in enumerate(outs[::-1][1:]):
                f = F.upsample(x_, scale_factor= 2**(i+1), mode='bilinear', align_corners=False)
                feature = torch.cat((feature, f), dim=1)
            return F.dropout(feature, p=0.5)
        elif self.out_type == 'last':
            return F.dropout(x, p=0.5)
        else:
            return tuple(outs)
