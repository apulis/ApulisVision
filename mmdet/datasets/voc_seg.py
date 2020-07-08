import os.path as osp
import numpy as np
from PIL import Image
import mmcv
from torch.utils.data import Dataset
from .builder import DATASETS
from .pipelines import Compose
from .utils import inv_mapping

@DATASETS.register_module()
class VOCSegmentation(Dataset):

    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 test_mode=False):
        self.data_root = data_root
        self.ann_file = ann_file,
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.test_mode = test_mode
        self.num_classes = len(self.CLASSES) + 1
        self.COLORS = self.get_class_colors()
        self.lable2color = {i:cat_id for i, cat_id in enumerate(self.COLORS)}
        self.rgb2label = inv_mapping(self.lable2color)

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
 
        # filter images too small
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # processing pipeline
        self.pipeline = Compose(pipeline)

    def get_class_colors(self):
        return [[0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128],
                [128, 0, 0], [128, 0, 128], [128, 128, 0],
                [128, 128, 128],
                [0, 0, 64], [0, 0, 192], [0, 128, 64],
                [0, 128, 192],
                [128, 0, 64], [128, 0, 192], [128, 128, 64],
                [128, 128, 192], [0, 64, 0], [0, 64, 128],
                [0, 192, 0],
                [0, 192, 128], [128, 64, 0], ]

    def load_annotations(self, ann_file):
        data_infos = []
        img_ids =  self.read_imglist(ann_file)
        for img_id in img_ids:
            filename = f'{img_id}.jpg'

            img_path = osp.join(self.img_prefix, '{}.jpg'.format(img_id))
            img = Image.open(img_path)
            width, height = img.size
            data_infos.append(
                dict(id=img_id, filename=filename, img_path=img_path, width=width, height=height))

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
        seg_path = osp.join(self.seg_prefix, '{}.png'.format(img_id))
        ann = dict(seg_path=seg_path)
        return ann


    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['mask_fields'] = []
        results['seg_fields'] = []


    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.data_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

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
                       h = img_info['height'],
                       w = img_info['width'],
                       rgb2label = self.rgb2label,
                       num_classes =  self.num_classes
                       )
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img_info = self.data_infos[idx]
        results = dict(img_path=img_info['img_path'], 
                       label_path=None,
                       img_id=img_info['img_id'],
                       h = img_info['height'],
                       w = img_info['width'],
                       rgb2label = self.rgb2label,
                       num_classes =  self.num_classes
                       )
        self.pre_pipeline(results)
        return self.pipeline(results)