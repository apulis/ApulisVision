import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss


@LOSSES.register_module
class CrossEntropyLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=False,
                 alpha=None,
                 reduction='mean',
                 loss_weight=1.0):
        super(CrossEntropyLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight
        if alpha is not None:
            if use_sigmoid:
                assert isinstance(alpha, (float, int))
                self.alpha = alpha
            else:
                assert isinstance(alpha, list)
                self.alpha = torch.Tensor(alpha)
        else:
            self.alpha = None

    def forward(self,
                inputs,
                targets,
                weight=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        targets = targets.long()
        if self.use_sigmoid:
            if self.alpha is not None:
                w = targets*self.alpha + (1-targets)*(1-self.alpha)
            else:
                w = None
            loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), weight=w, reduction='none')
        else:
            loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')

        if weight is not None:
            weight = weight.float()
        loss = weight_reduce_loss(
            loss, weight=weight, reduction=reduction)

        return self.loss_weight * loss

