import os
import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import auto_fp16, force_fp32, mask_target
from mmcv.cnn import ConvModule, constant_init, normal_init, build_conv_layer,  build_upsample_layer
from mmdet.ops.carafe import CARAFEPack
from objdet.models.builder import HEADS
from objdet.models.builder import build_loss


class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1).contiguous() # batch x hw x c
        probs = F.softmax(self.scale * probs, dim=2)# batch x k x hw
        ocr_context = torch.matmul(probs, feats).permute(0, 2, 1).contiguous().unsqueeze(3)# batch x k x c
        return ocr_context


class _ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''
    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            ConvModule(
                in_channels,
                key_channels,
                1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                inplace=True),
            ConvModule(
                key_channels,
                key_channels,
                1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                inplace=True),
        )
        self.f_object = nn.Sequential(
            ConvModule(
                in_channels,
                key_channels,
                1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                inplace=True),
            ConvModule(
                key_channels,
                key_channels,
                1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                inplace=True),
        )
        self.f_down = ConvModule(
                in_channels,
                key_channels,
                1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                inplace=True)
        self.f_up = ConvModule(
                key_channels,
                in_channels,
                1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                inplace=True)

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1).contiguous()
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1).contiguous()

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=ALIGN_CORNERS)

        return context


class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(ObjectAttentionBlock2D, self).__init__(in_channels,
                                                     key_channels,
                                                     scale,
                                                     conv_cfg,
                                                     norm_cfg)


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """
    def __init__(self,
                 in_channels,
                 key_channels,
                 out_channels,
                 scale=1,
                 drop_out=0.1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels,
                                                           key_channels,
                                                           scale,
                                                           conv_cfg,
                                                           norm_cfg)
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            ConvModule(
                _in_channels,
                out_channels,
                1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                inplace=True),
            nn.Dropout2d(drop_out)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output


@HEADS.register_module
class OCRHead(nn.Module):
    def __init__(self,
                 in_channel,
                 in_channel_aux,
                 inter_channel_aux,
                 ocr_mid_channels=512,
                 ocr_key_channels=256,
                 drop_out=0.05,
                 num_classes=81,
                 upsample_cfg=dict(type='bilinear', scale_factor=2, upsamle_size=None),
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_mask=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 loss_mask_aux=None,
                 loss_context=None,
                 loss_edge=None):
        super(OCRHead, self).__init__()
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
        self.in_channel_aux = in_channel_aux
        self.num_classes = num_classes
        self.upsample_method = self.upsample_cfg.get('type')
        self.scale_factor = self.upsample_cfg.pop('scale_factor', None)
        self.upsample_size = self.upsample_cfg.pop('upsample_size', None)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False

        self.loss_mask = build_loss(loss_mask) if loss_mask is not None else None
        self.loss_mask_aux = build_loss(loss_mask_aux) if loss_mask_aux is not None else None
        self.loss_context = build_loss(loss_context) if loss_context is not None else None
        self.loss_edge = build_loss(loss_edge) if loss_edge is not None else None

        self.conv3x3_ocr = ConvModule(
                in_channel,
                ocr_mid_channels,
                3,
                stride=1,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                inplace=True)
        self.ocr_gather_head = SpatialGather_Module(num_classes)

        self.ocr_distri_head = SpatialOCR_Module(in_channels=ocr_mid_channels,
                                                 key_channels=ocr_key_channels,
                                                 out_channels=ocr_mid_channels,
                                                 scale=1,
                                                 drop_out=drop_out,
                                                 )
        self.conv_logits = build_conv_layer(
            conv_cfg,
            ocr_mid_channels,
            num_classes,
            1,
            bias=True)
        self.conv_logits_aux = nn.Sequential(
            ConvModule(
                in_channel_aux,
                inter_channel_aux,
                1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                inplace=True),
            build_conv_layer(
                conv_cfg,
                inter_channel_aux,
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
    def forward(self, feat):
        if len(feat) == 1:
            x = feat[0]
            c = feat[0]
        else:
            x = feat[-1]
            c = feat[-2]
        mask_pred = []

        # ocr
        c = self.conv_logits_aux(c)
        # compute contrast feature
        x = self.conv3x3_ocr(x)

        context = self.ocr_gather_head(x, c)
        x = self.ocr_distri_head(x, context)

        x = self.conv_logits(x)

        if self.upsample is not None:
            mask_pred.append(self.upsample(c))
            mask_pred.append(self.upsample(x))
        else:
            mask_pred.append(c)
            mask_pred.append(x)
        return tuple(mask_pred)

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, mask_targets, valid_pixels=None):
        loss = dict()
        if self.loss_mask is not None:
            loss_mask = self.loss_mask(mask_pred[1], mask_targets, valid_pixels)
            loss['loss_mask'] = loss_mask
        if self.loss_mask_aux is not None:
            loss_mask_aux = self.loss_mask_aux(mask_pred[1], mask_targets, valid_pixels)
            loss['loss_mask_aux'] = loss_mask_aux
        if self.loss_context is not None:
            loss_context = self.loss_context(mask_pred[0], mask_targets, valid_pixels)
            loss['loss_context'] = loss_context
        if self.loss_edge is not None:
            loss_edge = self.loss_edge(mask_pred[1], mask_targets, valid_pixels)
            loss['loss_edge'] = loss_edge
        return loss


