import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from .adaptive_pool2d import SelectAdaptivePool2d
from ..builder import CLASSIFIERS,  build_backbone


def accuracy(pred, target, topk=1):
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    _, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / pred.size(0)))
    return res[0] if return_single else res


class Accuracy(nn.Module):

    def __init__(self, topk=(1, )):
        super().__init__()
        self.topk = topk

    def forward(self, pred, target):
        return accuracy(pred, target, self.topk)


@CLASSIFIERS.register_module
class Classifier(nn.Module):
    """Base class for Image classification.

    Image classification typically consisting of a Backbone network and a fchead.
    """
    def __init__(self, backbone, num_classes=1000, global_pool='avg', pretrained=None):
        super(Classifier, self).__init__()
        self.backbone = build_backbone(backbone)
        self.num_features = (self.backbone.feat_dim)
        self.init_weights(pretrained=pretrained)
        self.num_classes = num_classes

        # Head (Pooling and Classifier)
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.fc = nn.Linear(self.num_features * self.global_pool.feat_mult(), self.num_classes)
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained=pretrained)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        return x

    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """
        Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def forward_train(self, img, img_metas, gt_labels,**kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            gt_labels (list[Tensor]): class indices corresponding to each box

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()
        x = self.extract_feat(img)
        output = x[-1]
        output = self.global_pool(output).flatten(1)
        output = self.fc(output)
        loss = self.criterion(output, gt_labels)
        losses['loss'] = loss
        losses['acc@1'] = accuracy(output, gt_labels)
        losses['acc@5'] = accuracy(output, gt_labels, topk=5)
        return losses

    def forward_test(self, img, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
        """
        losses = dict()
        x = self.extract_feat(img)
        output = x[-1]
        output = self.global_pool(output).flatten(1)
        output = self.fc(output)
        _, predicted = torch.max(output, 1)
        return predicted.cpu().numpy()