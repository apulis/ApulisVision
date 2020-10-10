import argparse

import mmcv
from mmcv import Config

BACKBONES_ = dict(
    resnet='ResNet',
    resnetv1c='ResNetV1c',
    resnetv1d='ResNetV1d',
    resnext='ResNeXt')
HEADS_ = dict(fcnhead='FCNHead')
OPTS_ = dict(sgd='SGD', adam='Adam')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--pipeline_config', help='train config file path')
    args = parser.parse_args()
    return args


def update_configs(input_cfg):
    my_cfg = dict()
    for index, mdict in enumerate(input_cfg):
        # input module
        name = mdict['id'].lower()
        # if name == 'input':
        if index == 0:
            if mdict['name'].lower() == 'ade20k':
                my_cfg['dataset_type'] = 'ADE20KDataset'
            if mdict['name'].lower() == 'cityscapes':
                my_cfg['dataset_type'] = 'CityscapesDataset'
            if mdict['name'].lower() == 'pascal_voc12':
                my_cfg['dataset_type'] = 'PascalVOCDataset'
            my_cfg['data'] = {}
            my_cfg['data']['dataset_name'] = mdict['name']
            for mconfig in mdict['config']:
                my_cfg['data'][mconfig['key']] = mconfig['value']
        # # backbone
        # if name == 'backbone':
        #     if mdict['name'].lower() in BACKBONES_:
        #         my_cfg['backbone'] = {}
        #         my_cfg['backbone']['type'] = BACKBONES_[mdict['name'].lower()]
        #         for mconfig in mdict['config']:
        #             my_cfg['backbone'][mconfig['key']] = mconfig['value']
        # # neck
        # # head 1
        # if name == 'decode_head':
        #     if mdict['name'].lower() in HEADS_:
        #         my_cfg['decode_head'] = {}
        #         my_cfg['decode_head']['type'] = HEADS_[mdict['name'].lower()]
        #         for mconfig in mdict['config']:
        #             my_cfg['decode_head'][mconfig['key']] = mconfig['value']
        # # head 2
        # if name == 'auxiliary_head':
        #     if mdict['name'].lower() in HEADS_:
        #         my_cfg['auxiliary_head'] = {}
        #         my_cfg['auxiliary_head']['type'] = HEADS_[
        #             mdict['name'].lower()]
        #         for mconfig in mdict['config']:
        #             my_cfg['auxiliary_head'][mconfig['key']] = mconfig['value']
        # # optimizer
        # if name == 'optimizer':
        #     if mdict['name'].lower() in OPTS_:
        #         my_cfg['optimizer'] = {}
        #         mconfig = mdict['config'][0]
        #         my_cfg['optimizer']['type'] = OPTS_[mdict['name'].lower()]
        #         my_cfg['optimizer']['lr'] = mconfig['value']
        # output
        # if name == 'output':
        #     my_cfg['runtime'] = {}
        #     for mconfig in mdict['config']:
        #         my_cfg['runtime'][mconfig['key']] = mconfig['value']
        if index == len(input_cfg)-1:
            my_cfg['runtime'] = {}
            for mconfig in mdict['config']:
                my_cfg['runtime'][mconfig['key']] = mconfig['value']
    return my_cfg


def merge_from_mycfg(my_cfg, cfg):
    # update model config
    if my_cfg.__contains__('backbone'):
        cfg.model.backbone.update(my_cfg['backbone'])
    elif my_cfg.__contains__('decode_head'):
        cfg.model.decode_head.update(my_cfg['decode_head'])
    elif my_cfg.__contains__('auxiliary_head'):
        cfg.model.auxiliary_head.update(my_cfg['auxiliary_head'])
    # update data config
    data_path = my_cfg['data']['data_path']
    cfg.data.train.data_root = data_path
    cfg.data.train.img_dir = 'leftImg8bit/train'
    cfg.data.train.ann_dir = 'gtFine/train'
    # val
    cfg.data.val.data_root = data_path
    cfg.data.val.img_dir = 'leftImg8bit/val'
    cfg.data.val.ann_dir = 'gtFine/val'
    # test
    cfg.data.test.data_root = data_path
    cfg.data.test.img_dir = 'leftImg8bit/val'
    cfg.data.test.ann_dir = 'gtFine/val'
    cfg.data.samples_per_gpu = my_cfg['runtime']['batch_size']
    # update optimizer
    if my_cfg.__contains__('optimizer'):
        cfg.optimizer.update(my_cfg['optimizer'])
    # update runtime
    cfg.total_iters = my_cfg['runtime']['total_iters']
    cfg.work_dir = my_cfg['runtime']['work_dir']
    return cfg


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    input_cfg = mmcv.load(args.pipeline_config)
    my_cfg = update_configs(input_cfg)
    print(my_cfg)
    new_cfg = merge_from_mycfg(my_cfg, cfg)
    print()
    print(new_cfg)


if __name__ == '__main__':
    main()
