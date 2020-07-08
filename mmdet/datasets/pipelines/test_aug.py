import mmcv

from ..builder import PIPELINES
from .compose import Compose


@PIPELINES.register_module
class MultiScaleFlipAug(object):

    def __init__(self, transforms, img_scale, flip=False):
        self.transforms = Compose(transforms)
        self.img_scale = img_scale if isinstance(img_scale,
                                                 list) else [img_scale]
        assert mmcv.is_list_of(self.img_scale, tuple)
        self.flip = flip

    def __call__(self, results):
        aug_data = []
        flip_aug = [False, True] if self.flip else [False]
        for scale in self.img_scale:
            for flip in flip_aug:
                _results = results.copy()
                _results['scale'] = scale
                _results['flip'] = flip
                data = self.transforms(_results)
                aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(transforms={}, img_scale={}, flip={})'.format(
            self.transforms, self.img_scale, self.flip)
        return repr_str


@PIPELINES.register_module
class MultiTestAug(object):

    def __init__(self, transforms, aug_list=None):
        self.transforms = Compose(transforms)
        self.aug_list = []
        if aug_list:
            for aug in aug_list:
                if isinstance(aug_list[0], list):
                    self.aug_list.append(aug)
                else:
                    self.aug_list.append([aug])

    def __call__(self, results):
        aug_data = []
        _results = results.copy()
        aug_data.append(self.transforms(_results))
        for aug in self.aug_list:
            augs = Compose(aug)
            _results = results.copy()
            _results = augs(_results)
            data = self.transforms(_results)
            aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(transforms={}, img_scale={}, flip={})'.format(
            self.transforms, self.img_scale, self.flip)
        return repr_str
