#!/usr/bin/python
from PIL import Image
import numpy as np
import json
import random
import os

from scipy.ndimage.morphology import distance_transform_edt

Image.MAX_IMAGE_PIXELS = None


def read_image(img_path, mode='image', n_channels=3, target_size=-1):
    if not os.path.isfile(img_path):
        print("File %s does not exist" % (img_path))
        return None
    if not img_path.split('.')[-1] in ['tif', 'png', 'jpg', 'jpeg', 'ppm', 'bmp']:
        print("File %s is not supported" % (img_path))
        return None

    img = Image.open(img_path)
    if n_channels != 1:
        if not img.mode in ['RGBA', 'RGB']:
            img = img.convert('RGBA')
        if n_channels == 3 and img.mode == 'RGBA':
            img = img.convert('RGB')
        elif n_channels == 4 and img.mode == 'RGB':
            img = img.convert('RGBA')

    if target_size > 0:
        #img = cv2.resize(img, dsize=(img.shape[1]//target_size, img.shape[0]//target_size), interpolation=cv2.INTER_NEAREST)
        if mode == 'image':
            img = img.resize((img.size[0] // target_size, img.size[1] // target_size), Image.BILINEAR)
        elif mode == 'label':
            img = img.resize((img.size[0] // target_size, img.size[1] // target_size), Image.NEAREST)
        else:
            print("Reading mode %s is not supported" % (mode))
            return None
    return np.array(img)


def label_color_map(num_classes):
    label2rgb = {}
    scale = 255 * 255 * 255 // num_classes
    for label in range(num_classes):
        temp = label * scale
        r = temp // (255*255)
        g = temp % (255*255) // 255
        b = temp % 255
        label2rgb[label] = (r, g, b)

    return label2rgb


def inv_mapping(map):
    inv_map = {}
    for key, value in map.items():
        if isinstance(value, list):
            value = tuple(value)
        assert value not in list(inv_map.keys()), 'Cannot inverse map: same value for different keys.'
        inv_map[value] = key

    return inv_map


def color2class(label_img_rgb, rgb2label):
    label_img_rgb = label_img_rgb.astype(np.uint32)
    cm2lbl = np.zeros(256 ** 3, dtype=np.uint16)
    for color, label in rgb2label.items():
        cm2lbl[(color[0] * 256 + color[1]) * 256 + color[2]] = label
    idx = (label_img_rgb[:, :, 0] * 256 +label_img_rgb[:, :, 1]) * 256 + label_img_rgb[:, :, 2]

    return np.array(cm2lbl[idx])


def label2mask(input, label2rgb):
    cmap = np.zeros((len(label2rgb), 3), dtype=np.uint8)
    for label, color in label2rgb.items():
        cmap[int(label), :] = list(color)

    return np.array(cmap[input])


def mask_to_onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    """
    _mask = [mask == (i + 1) for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)


def onehot_to_mask(mask):
    """
    Converts a mask (K,H,W) to (H,W)
    """
    _mask = np.argmax(mask, axis=0)
    _mask[_mask != 0] += 1
    return _mask


def onehot_to_multiclass_edges(mask, radius, num_classes):
    """
    Converts a segmentation mask (K,H,W) to an edgemap (K,H,W)

    """
    if radius < 0:
        return mask

    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

    channels = []
    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(1.0 - mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        dist = (dist > 0).astype(np.uint8)
        channels.append(dist)

    return np.array(channels)


def onehot_to_binary_edges(mask, radius, num_classes):
    """
    Converts a segmentation mask (K,H,W) to a binary edgemap (H,W)

    """

    if radius < 0:
        return mask

    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

    edgemap = np.zeros(mask.shape[1:])

    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(1.0 - mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edgemap += dist
    #edgemap = np.expand_dims(edgemap, axis=0)
    edgemap = (edgemap > 0).astype(np.uint8)
    return edgemap
