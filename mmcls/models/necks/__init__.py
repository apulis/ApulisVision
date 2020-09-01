from .adaptive_avgmaxpool2d import AdaptiveAvgMaxPool2d
from .adaptive_catavgmaxpool2d import AdaptiveCatAvgMaxPool2d
from .gap import GlobalAveragePooling

__all__ = [
    'GlobalAveragePooling', 'AdaptiveAvgMaxPool2d', 'AdaptiveCatAvgMaxPool2d'
]
