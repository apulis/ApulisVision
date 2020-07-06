#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""ImageNet dataset."""

import os
import re
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader,DatasetFolder
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score


def build_dataset(cfg, default_args=None):
    from .dataset_wrappers import (ConcatDataset, RepeatDataset,
                                   ClassBalancedDataset)
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'ClassBalancedDataset':
        dataset = ClassBalancedDataset(
            build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    else:
        dataset = build_cls_dataset(cfg)
    return dataset


def build_cls_dataset(cfg=None):
    """
        Args:
            data_dir: image data root
            ann_file:
            transform:
        Returns:
    """
    train_transform =  transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    val_transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    if cfg.ann_file is not None:
        datasets = MultiClassDataset(
                ann_file = cfg.ann_file,
                img_prefix = cfg.img_prefix,
                transform = train_transform if cfg.train_mode else val_transform,
            )
    else:
        datasets = ImageFolder(os.path.join(cfg.img_prefix),
                transform = train_transform if cfg.train_mode else val_transform,
            )
    return datasets


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples
        self.CLASSES = self.classes

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        result = dict(img = sample, gt_labels=target)
        return result

    def get_ann_info(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        return target


    def format_results(self, results, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list): Testing results of the dataset.
        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json filepaths, tmp_dir is the temporal directory created
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        outputs = []
        for res in results:
            outputs.extend(res.tolist())
        return outputs


    def evaluate(self,
                results,
                metric='acc',
                logger=None):
        """Evaluate the dataset.
        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['acc', 'acc5', 'recall', 'f1']
        if metric not in allowed_metrics:
            raise KeyError('metric {} is not supported'.format(metric))
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        results = self.format_results(results)
        eval_results = {}
        
        eval_results['acc_micro'] = precision_score(annotations, results, average='micro')  
        eval_results['acc_macro'] = precision_score(annotations, results, average='macro')  
        return eval_results

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


class MultiClassDataset(Dataset):
    """
    Custom Image Classification datasets, 
    A generic data loader where the images are arranged your data in this  way

        /data/train/images/01.jpg,0
        /data/train/images/02.jpg,1
        /data/train/images/03.jpg,0
        /data/train/images/04.jpg,1
        
    Args:

        ann_file: image label file.
        img_prefix:  images prefix.
        ...
        
    """

    def __init__(
        self,
        ann_file,
        img_prefix='',
        loader=default_loader,
        transform=None,
        target_transform=None):

        super(MultiClassDataset, self).__init__()
        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform

        self.labels_file = self.ann_file
        self.samples, self.CLASSES = _make_dataset(self.labels_file)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        path = os.path.join(self.img_prefix, path)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        result = dict(img = sample, gt_labels=target)

        return result

    def get_ann_info(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        return target


    def format_results(self, results, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list): Testing results of the dataset.
        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json filepaths, tmp_dir is the temporal directory created
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        outputs = []
        for res in results:
            outputs.extend(res.tolist())
        return outputs


    def evaluate(self,
                results,
                metric='acc',
                logger=None):
        """Evaluate the dataset.
        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['acc', 'acc5', 'recall', 'f1']
        if metric not in allowed_metrics:
            raise KeyError('metric {} is not supported'.format(metric))
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        results = self.format_results(results)
        eval_results = {}
        eval_results['acc_micro'] = precision_score(annotations, results, average='micro')  # 微平均，精确率
        eval_results['acc_macro'] = precision_score(annotations, results, average='macro')  # 微平均，精确率
        return eval_results


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def _make_dataset(labels_file):
    """
    Args:
        labels_file:
    Returns:
    """
    samples = []
    classes = []
    with open(labels_file) as labels_file_open:
        for line in labels_file_open.readlines():
            parts = [part.strip() for part in line.split(",")]
            path = parts[0]
            cls_name = parts[1]
            target = int(parts[2])
            samples.append((path, target))
            if cls_name not in classes:
                classes.append(cls_name)
    return samples, classes


def _make_resample(samples):
    """
    Args:
        samples:
    Returns:
    """
    class_to_paths = {}
    for path, target in samples:
        if target in class_to_paths:
            class_to_paths[target].append(path)
        else:
            class_to_paths[target] = [path]
    ret = []
    max_len = max([len(paths) for paths in class_to_paths.values()])
    for target, paths in class_to_paths.items():
        copy_times = int(round(max_len / len(paths)))
        for path in paths:
            for _ in range(copy_times):
                ret.append((path, target))
    return ret


def _stats_samples_distribution(samples):
    """
    Args:
        samples:
    Returns:
    """
    class_to_nums = {}
    for path, target in samples:
        if target in class_to_nums:
            class_to_nums[target] += 1
        else:
            class_to_nums[target] = 1
    return class_to_nums