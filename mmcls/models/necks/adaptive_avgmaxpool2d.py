import torch
import torch.nn as nn

from ..builder import NECKS


@NECKS.register_module()
class AdaptiveAvgMaxPool2d(nn.Module):

    def __init__(self):
        super(AdaptiveAvgMaxPool2d, self).__init__()
        self.avgmaxpool = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self):
        pass

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            outs = tuple([self.avgmaxpool(x) for x in inputs])
            outs = tuple(
                [out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, torch.Tensor):
            outs = self.avgmaxpool(inputs)
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs
