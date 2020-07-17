import numpy as np
import torch
from PIL import Image


def merge_aug_segmaps(results, num_classes, mode='average', aug_weights=None, return_proba=False):
    if aug_weights:
        assert isinstance(aug_weights, (list, tuple)) and len(aug_weights) == len(results)
    assert mode in ['average', 'vote']
    if mode == 'average':
        if aug_weights is not None:
            merge_results = np.average(np.array(results), axis=0, weights=np.array(aug_weights))
        else:
            merge_results = np.mean(results, axis=0)
        if return_proba:
            #seg_results = np.sqrt(np.square(merge_results) / np.sum(np.square(merge_results), axis=1))
            seg_results = merge_results
        else:
            seg_results = np.argmax(merge_results, axis=1)
    elif mode == 'vote':
        assert not return_proba
        merge_results = np.array(results)
        if merge_results.ndim > 4:
            merge_results = np.argmax(merge_results, axis=2)
        def func(data):
            return np.bincount(data, minlength=num_classes)
        count = np.apply_along_axis(func, 0, merge_results)
        seg_results = np.argmax(count, axis=0)
    else:
        raise NotImplementedError
    return seg_results


def recover_mask(imgs, img_metas):
    #TODO: realize with numpy
    new_imgs = []
    for i in range(imgs.shape[0]):
        temp_img = []
        for j in range(imgs.shape[1]):
            img = Image.fromarray(imgs[i][j])
            if img_metas[i].get('rotate'):
                img = img.rotate(360 - img_metas[i]['rotate'], Image.NEAREST)
            if img_metas[i].get('horizontal_flip'):
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if img_metas[i].get('vertical_flip'):
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img = img.resize(img_metas[i]['ori_shape'], Image.NEAREST)
            temp_img.append(np.array(img))
        new_imgs.append(np.array(temp_img))
    return np.array(new_imgs)