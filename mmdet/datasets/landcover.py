#!/usr/bin/python
import random
import mmcv
import os.path as osp
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from .builder import DATASETS
from .pipelines import Compose
from mmdet.datasets.utils.utils import inv_mapping


@DATASETS.register_module
class DGLandcoverDataset(Dataset):
    """A generic data loader where the images are arranged in this way: ::

    images:
        root/image/xxx.jpg
        root/image/xxy.jpg
        root/image/xxz.jpg

    labels:
        root/label/xxx.png
        root/label/xxy.png
        root/label/xxz.png

    Args:
        data_path (string): Root directory path.
        label_path (string): Root directory path
        pipeline (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``

    """

    CLASSES = ['Unknown', 'Urban land', 'Agriculture land', 'Range land', 'Forest land', 'Water', 'Barren land']
    COLORS = {0: [0, 0, 0], 1: [0, 255, 255], 2: [255, 255, 0], 3: [255, 0, 255], 4: [0, 255, 0], 5: [0, 0, 255], 6: [255, 255, 255]}
    HEIGHT = 2448
    WIDTH = 2448

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root,
                 img_prefix='',
                 seg_prefix='',
                 n_channels=3,
                 tile_size=1500,
                 stride=1500,
                 test_mode=False):

        self.data_root = data_root
        self.ann_file = ann_file,
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.test_mode = test_mode
        self.n_channels = n_channels
        self.tile_size = tile_size
        self.stride = stride
        self.num_classes = len(self.CLASSES)
        self.rgb2label = inv_mapping(self.COLORS)

        # join paths if data_root is specified
        if self.data_root is not None:
            # if not osp.isabs(self.ann_file):
            #     self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)

        # load annotations (and proposals)
        self.data_infos = self.load_annotations(self.ann_file)
 
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # processing pipeline
        self.pipeline = Compose(pipeline)


    def load_annotations(self, ann_file):
        data_infos = []
        img_ids =  self.read_imglist(ann_file)
        for img_id in img_ids:
            filename = f'{img_id}.jpg'
            img_path = osp.join(self.img_prefix, '{}.jpg'.format(img_id))
            if not self.test_mode:
                for i in range(0, self.HEIGHT, self.stride):
                    for j in range(0, self.WIDTH, self.stride):
                        i_idx = min(i, self.HEIGHT - self.tile_size)
                        j_idx = min(j, self.WIDTH - self.tile_size)
                        data_infos.append(dict(id=img_id, filename=filename, w=i_idx, h=j_idx))
                        if (j + self.stride) >= self.HEIGHT:
                            break
                    if (i + self.stride) >= self.WIDTH:
                        break
            else:
                for i in range(0, self.HEIGHT, self.tile_size):
                    for j in range(0, self.WIDTH, self.tile_size):
                        i_idx = min(i, self.HEIGHT - self.tile_size)
                        j_idx = min(j, self.WIDTH - self.tile_size)
                        data_infos.append(dict(id=img_id, filename=filename, img_path=img_path, w=i_idx, h=j_idx))
        return data_infos


    def read_imglist(self, imglist):
        filelist = []
        with open(imglist[0], 'r') as fd:
            for line in fd:
                filelist.append(line.strip())
        return filelist

    def get_ann_info(self, idx):
        img_info = self.data_infos[idx]
        img_id = img_info['id']

        seg_map =  f'{img_id}.png'
        seg_path = osp.join(self.seg_prefix, '{}.png'.format(img_id))
        ann = dict(seg_map=seg_map, seg_path=seg_path)
        return ann

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['mask_fields'] = []
        results['seg_fields'] = []
    

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_path=img_info['img_path'], 
                       label_path=ann_info['seg_path'],
                       img_id=img_info['id'],
                       full_shape=(self.HEIGHT, self.WIDTH),
                       ori_shape=(self.tile_size, self.tile_size),
                       tile_size=self.tile_size,
                       img_shape=(self.tile_size, self.tile_size),
                       h = img_info['h'],
                       w = img_info['w'],
                       rgb2label = self.rgb2label,
                       num_classes =  self.num_classes
                       )
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img_info = self.data_infos[idx]
        results = dict(img_path=img_info['img_path'], 
                       label_path=None,
                       img_id=img_info['id'],
                       full_shape=(self.HEIGHT, self.WIDTH),
                       ori_shape=(self.tile_size, self.tile_size),
                       tile_size=self.tile_size,
                       img_shape=(self.tile_size, self.tile_size),
                       h = img_info['h'],
                       w = img_info['w'],
                       rgb2label = self.rgb2label,
                       num_classes =  self.num_classes
                       )
        self.pre_pipeline(results)
        return self.pipeline(results)