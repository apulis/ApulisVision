import argparse
import os

import mmcv
from mmcv import Config

BACKBONES_ = dict(
    resnet='ResNet',
    resnext='ResNeXt',
    seresnet='SEResNet',
    serenext='SEResNeXt')
NECKS_ = dict(
    globalaveragepooling='GlobalAveragePooling',
    adaptivecatavgmaxpool2d='AdaptiveCatAvgMaxPool2d',
    adaptiveavgmaxpool2d='AdaptiveAvgMaxPool2d')
HEADS_ = dict(linearclshead='LinearClsHead', num_classes=1000, topk=(1, 5))
OPTS_ = dict(sgd='SGD', adam='Adam')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--pipeline_config', help='train config file path')
    args = parser.parse_args()
    return args


def update_configs(input_cfg):
    my_cfg = dict()
    for mdict in input_cfg:
        # input module
        name = mdict['id'].lower()
        if name == 'input':
            if mdict['name'].lower() == 'mnist':
                my_cfg['dataset_type'] = 'MNIST'
            if mdict['name'].lower() == 'cifar10':
                my_cfg['dataset_type'] = 'CIFAR10'
            else:
                my_cfg['dataset_type'] = 'ImageNet'
            my_cfg['data'] = {}
            my_cfg['data']['dataset_name'] = mdict['name']
            for mconfig in mdict['config']:
                my_cfg['data'][mconfig['key']] = mconfig['value']
        # backbone
        if name == 'backbone':
            if mdict['name'].lower() in BACKBONES_:
                my_cfg['backbone'] = {}
                my_cfg['backbone']['type'] = BACKBONES_[mdict['name'].lower()]
                for mconfig in mdict['config']:
                    my_cfg['backbone'][mconfig['key']] = mconfig['value']
        # neck
        if name == 'neck':
            if mdict['name'].lower() in NECKS_:
                my_cfg['neck'] = {}
                my_cfg['neck']['type'] = NECKS_[mdict['name'].lower()]
        # head
        if name == 'head':
            if mdict['name'].lower() in HEADS_:
                my_cfg['head'] = {}
                my_cfg['head']['type'] = HEADS_[mdict['name'].lower()]
                for mconfig in mdict['config']:
                    my_cfg['head'][mconfig['key']] = mconfig['value']
                if my_cfg['head']['num_classes'] > 5:
                    my_cfg['head']['topk'] = (1, 5)
                else:
                    my_cfg['head']['topk'] = (1, )
        # optimizer
        if name == 'optimizer':
            if mdict['name'].lower() in OPTS_:
                my_cfg['optimizer'] = {}
                mconfig = mdict['config'][0]
                my_cfg['optimizer']['type'] = OPTS_[mdict['name'].lower()]
                my_cfg['optimizer']['lr'] = mconfig['value']
        # output
        if name == 'output':
            my_cfg['runtime'] = {}
            for mconfig in mdict['config']:
                my_cfg['runtime'][mconfig['key']] = mconfig['value']
    return my_cfg


def merge_from_mycfg(my_cfg, cfg):
    # update model config
    cfg.model.backbone.update(my_cfg['backbone'])
    cfg.model.neck.update(my_cfg['neck'])
    cfg.model.head.update(my_cfg['head'])
    # update data config
    data_path = my_cfg['data']['data_path']
    dataset_name = my_cfg['data']['dataset_name']
    my_cfg['train_data'] = os.path.join(data_path, dataset_name)
    my_cfg['test_data'] = os.path.join(data_path, dataset_name)
    cfg.data.train.data_prefix = my_cfg['train_data']
    cfg.data.val.data_prefix = my_cfg['test_data']
    cfg.data.test.data_prefix = my_cfg['test_data']
    cfg.data.samples_per_gpu = my_cfg['runtime']['batch_size']
    # update optimizer
    cfg.optimizer.update(my_cfg['optimizer'])
    # update runtime
    cfg.total_epochs = my_cfg['runtime']['total_epochs']
    cfg.work_dir = my_cfg['runtime']['work_dir']
    return cfg


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    print(type(cfg))
    input_cfg = mmcv.load(args.pipeline_config)
    my_cfg = update_configs(input_cfg)
    print(my_cfg)
    new_cfg = merge_from_mycfg(my_cfg, cfg)
    print()
    print(new_cfg)


if __name__ == '__main__':
    main()
