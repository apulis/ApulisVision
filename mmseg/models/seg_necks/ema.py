import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import constant_init, kaiming_init
import math

from mmdet.core import auto_fp16
from mmcv.cnn import ConvModule, constant_init, normal_init, build_conv_layer, build_norm_layer
from ..builder import NECKS


@NECKS.register_module
class EMA(nn.Module):
    def __init__(self,
                 in_channels,
                 reduction_channels,
                 k,
                 stage_num=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(EMA, self).__init__()
        self.stage_num = stage_num

        mu = torch.Tensor(1, reduction_channels, k)
        mu.normal_(0, math.sqrt(2. / k))  # Init with Kaiming Norm.
        mu = self._l2norm(mu, dim=1)
        self.register_buffer('mu', mu)

        self.conv0 = ConvModule(in_channels,
                                reduction_channels,
                                kernel_size=3,
                                padding=1,
                                bias=False,
                                conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg)
        self.conv1 = build_conv_layer(conv_cfg, reduction_channels, reduction_channels, 1)
        self.conv2 = nn.Sequential(
            build_conv_layer(conv_cfg, reduction_channels, reduction_channels, 1, bias=False),
            build_norm_layer(norm_cfg, reduction_channels)[1])

    def init_weights(self):
        for m in self.modules():
            #if isinstance(m, nn.Conv2d):
                #xavier_init(m, distribution='uniform')
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

    def forward(self, x):
        x = self.conv0(x)
        idn = x
        # The first 1x1 conv
        x = self.conv1(x)

        # The EM Attention
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)  # b * c * n
        mu = self.mu.repeat(b, 1, 1)  # b * c * k
        with torch.no_grad():
            for i in range(self.stage_num):
                x_t = x.permute(0, 2, 1)  # b * n * c
                z = torch.bmm(x_t, mu)  # b * n * k
                z = F.softmax(z, dim=2)  # b * n * k
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
                mu = torch.bmm(x, z_)  # b * c * k
                mu = self._l2norm(mu, dim=1)

        z_t = z.permute(0, 2, 1)  # b * k * n
        x = mu.matmul(z_t)  # b * c * n
        x = x.view(b, c, h, w)  # b * c * h * w
        x = F.relu(x, inplace=True)

        # The second 1x1 conv
        x = self.conv2(x)
        x = x + idn
        x = F.relu(x, inplace=True)

        return x, mu

    def _l2norm(self, inp, dim):
        '''Normlize the inp tensor with l2-norm.

        Returns a tensor where each sub-tensor of input along the given dim is
        normalized such that the 2-norm of the sub-tensor is equal to 1.

        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.

        Returns:
            (tensor) The normalized tensor.
        '''
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))
