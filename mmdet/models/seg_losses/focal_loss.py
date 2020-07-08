import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ..builder import LOSSES
from .utils import weight_reduce_loss


@LOSSES.register_module
class FocalSegLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=False,
                 gamma=2.0,
                 alpha=None,
                 reduction='mean',
                 loss_weight=1.0):
        super(FocalSegLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
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
                input,
                target,
                weight=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        target = target.long()
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        if self.use_sigmoid:
            pred_sigmoid = pred.sigmoid()
            target = target.type_as(pred)
            pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
            focal_weight = (self.alpha * target + (1 - self.alpha) *
                            (1 - target)) * pt.pow(gamma)
            loss = F.binary_cross_entropy_with_logits(
                pred, target, reduction='none') * focal_weight

        else:
            logpt = F.log_softmax(input)
            logpt = logpt.gather(1, target)
            logpt = logpt.view(-1)
            pt = Variable(logpt.data.exp())

            if self.alpha is not None:
                if self.alpha.type() != input.data.type():
                    self.alpha = self.alpha.type_as(input.data)
                at = self.alpha.gather(0, target.data.view(-1))
                logpt = logpt * Variable(at)

            loss = -1 * (1 - pt) ** self.gamma * logpt

        if weight is not None:
            weight = weight.float()
            weight = weight.view(-1)
        loss = weight_reduce_loss(loss, weight, reduction)
        return self.loss_weight * loss

