"""Custom losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse
from ..builder import LOSSES

# --------------------------- LOVASZ HINGE LOSSES ---------------------------
@LOSSES.register_module
class LovaszHinge(nn.Module):
    def __init__(self,
                 per_image=False,
                 ignore=None,
                 loss_weight=1.0):
        super(LovaszHinge, self).__init__()
        self.per_image = per_image
        self.ignore = ignore
        self.loss_weight = loss_weight

    def forward(self, inputs, targets, valid_pixels=None):
        probas = F.sigmoid(inputs)
        if self.per_image:
            loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), self.ignore, valid_pixels))
                        for log, lab in zip(probas, targets))
        else:
            loss = lovasz_hinge_flat(*flatten_binary_scores(probas, targets, self.ignore, valid_pixels))
        return self.loss_weight * loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None, valid_pixels=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    valid_pixels = valid_pixels.view(-1)
    if (ignore is None) and (valid_pixels is None):
        return scores, labels
    if ignore is None:
        valid = (valid_pixels > 0)
    elif valid_pixels is None:
        valid = (labels != ignore)
    else:
        valid = ((labels != ignore) * (valid_pixels > 0))
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


# --------------------------- LOVASZ SOFTMAX LOSSES ---------------------------
@LOSSES.register_module
class LovaszSoftmax(nn.Module):
    def __init__(self,
                 classes='present',
                 per_image=False,
                 ignore=None,
                 loss_weight=1.0):
        super(LovaszSoftmax, self).__init__()
        self.per_image = per_image
        self.classes = classes
        self.ignore = ignore
        self.loss_weight = loss_weight

    def forward(self, inputs, targets, valid_pixels=None):
        probas = F.softmax(inputs, dim=1)
        if self.per_image:
            loss = mean(
                lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), self.ignore, valid_pixels), classes=self.classes)
                for prob, lab in zip(probas, targets))
        else:
            loss = lovasz_softmax_flat(*flatten_probas(probas, targets, self.ignore, valid_pixels), classes=self.classes)
        return self.loss_weight * loss


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None, valid_pixels=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if (ignore is None) and (valid_pixels is None):
        return probas, labels
    if ignore is None:
        valid_pixels = valid_pixels.view(-1)
        valid = (valid_pixels > 0)
    elif valid_pixels is None:
        valid = (labels != ignore)
    else:
        valid_pixels = valid_pixels.view(-1)
        valid = ((labels != ignore) * (valid_pixels > 0))
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n
