import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import NECKS




@NECKS.register_module()
class AdaptiveCatAvgMaxPool2d(nn.Module):
    def __init__(self):
        super(AdaptiveCatAvgMaxPool2d, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

    def init_weights(self):
        pass

    def forward(self, inputs):
        if isinstance(inputs, torch.Tensor):
            x_avg = self.avgpool(inputs)
            x_max = self.maxpool(inputs)
            outs = torch.cat((x_avg, x_max), 1)
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs