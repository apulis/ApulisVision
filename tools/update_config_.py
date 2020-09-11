import argparse
import os

import mmcv
from mmcv import Config

BACKBONES_ = dict(
    resnet='ResNet',
    resnext='ResNeXt',
    seresnet='SEResNet',
    serenext='SEResNeXt')
GAPS_ = dict(
    globalaveragepooling='GlobalAveragePooling',
    adaptivecatavgmaxpool2d='AdaptiveCatAvgMaxPool2d')
HEADS_ = dict(linearclshead='LinearClsHead', num_classes=1000, topk=(1, 5))
OPTS_ = dict(sgd='SGD', adam='Adam')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--pipeline_config', help='train config file path')
    args = parser.parse_args()
    return args


def update_configs(input_cfg):
    my_cfg = dict()
    input_cfg = input_cfg['nodes']
    for mdict in input_cfg:
        # input module
        name = mdict.id.lower()
        if name == 'input':
            if mdict.name.lower() == 'mnist':
                my_cfg['dataset_type'] = 'MNIST'
            if mdict.name.lower() == 'cifar10':
                my_cfg['dataset_type'] = 'CIFAR10'
            else:
                my_cfg['dataset_type'] = 'ImageNet'
            my_cfg['data'] = {}
            my_cfg['data'].update(mdict.config)
            my_cfg['data']['dataset_name'] = mdict.name
        # backbone
        if name == 'backbone':
            if mdict.name.lower() in BACKBONES_:
                my_cfg['backbone'] = {}
                my_cfg['backbone']['type'] = BACKBONES_[mdict.name.lower()]
            my_cfg['backbone'].update(mdict.config)
        # neck
        if name == 'neck':
            if mdict.name.lower() in GAPS_:
                my_cfg['neck'] = {}
                my_cfg['neck']['type'] = GAPS_[mdict.name.lower()]
                my_cfg['neck'].update(mdict.config)
        # head
        if name == 'head':
            if mdict.name.lower() in HEADS_:
                my_cfg['head'] = {}
                my_cfg['head']['type'] = HEADS_[mdict.name.lower()]
                my_cfg['head'].update(mdict.config)
                if my_cfg['head']['num_classes'] > 5:
                    my_cfg['head']['topk'] = (1, 5)
                else:
                    my_cfg['head']['topk'] = (1, )
        # optimizer
        if name == 'optimizer':
            if mdict.name.lower() in OPTS_:
                my_cfg['optimizer'] = {}
                my_cfg['optimizer']['type'] = OPTS_[mdict.name.lower()]
                my_cfg['optimizer'].update(mdict.config)
        # output
        if name == 'output':
            my_cfg['runtime'] = {}
            my_cfg['runtime'].update(mdict.config)
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
    cfg.data.samples_per_gpu = my_cfg['runtime']['batch_size']
    # update optimizer
    cfg.optimizer.update(my_cfg['optimizer'])
    # update runtime
    cfg.total_epochs = my_cfg['runtime']['total_epochs']
    cfg.work_dir = my_cfg['runtime']['work_dir']
    return cfg


def main():
    args = parse_args()
    output_dict = {}
    cfg = Config.fromfile(args.config)
    input_cfg = mmcv.load(args.pipeline_config)
    output_dict['nodes'] = input_cfg
    mmcv.dump(output_dict, file='panel.json', file_format='json')
    input_cfg = Config.fromfile('panel.json')
    my_cfg = update_configs(input_cfg)
    print(my_cfg)
    new_cfg = merge_from_mycfg(my_cfg, cfg)
    print()
    print(new_cfg)


if __name__ == '__main__':
    main()
