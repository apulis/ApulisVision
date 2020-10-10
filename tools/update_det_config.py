import argparse
import os

import mmcv
from mmcv import Config

BACKBONES_ = dict(resnet='ResNet', resnext='ResNeXt')
NECKS_ = dict(fpn='FPN', pafpn='PAFPN')
RPN_HEADS_ = dict(rpnhead='RPNHead')
ROI_HEAD_ = dict(standardroihead='StandardRoIHead')
OPTS_ = dict(sgd='SGD', adam='Adam')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--pipeline_config', help='train config file path')
    args = parser.parse_args()
    return args


def update_configs(input_cfg):
    """load panel file."""
    my_cfg = dict()
    for index, mdict in enumerate(input_cfg):
        # input module
        name = mdict['id'].lower()
        # if name == 'input':
        if index == 0:
            if mdict['name'].lower() == 'coco':
                my_cfg['dataset_type'] = 'CocoDataset'
            elif mdict['name'].lower() == 'voc':
                my_cfg['dataset_type'] = 'VOCDataset'
            elif mdict['name'].lower() == 'cityscapes':
                my_cfg['dataset_type'] = 'CityscapesDataset'
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
        # if name == 'neck':
        #     if mdict['name'].lower() in NECKS_:
        #         my_cfg['neck'] = {}
        #         my_cfg['neck']['type'] = NECKS_[mdict['name'].lower()]
        # # head
        # if name == 'rpn_head':
        #     if mdict['name'].lower() in RPN_HEADS_:
        #         my_cfg['rpn_head'] = {}
        #         my_cfg['rpn_head']['type'] = RPN_HEADS_[mdict['name'].lower()]
        # if name == 'roi_head':
        #     if mdict['name'].lower() in ROI_HEAD_:
        #         my_cfg['roi_head'] = {}
        #         my_cfg['roi_head']['type'] = ROI_HEAD_[mdict['name'].lower()]
        # # optimizer
        # if name == 'optimizer':
        #     if mdict['name'].lower() in OPTS_:
        #         my_cfg['optimizer'] = {}
        #         mconfig = mdict['config'][0]
        #         my_cfg['optimizer']['type'] = OPTS_[mdict['name'].lower()]
        #         my_cfg['optimizer']['lr'] = mconfig['value']
        # # output
        # if name == 'output':
        elif index == len(input_cfg)-1:
            my_cfg['runtime'] = {}
            for mconfig in mdict['config']:
                my_cfg['runtime'][mconfig['key']] = mconfig['value']
    return my_cfg


def merge_from_mycfg(my_cfg, cfg):
    # update model config
    if my_cfg.__contains__('backbone'):
        cfg.model.backbone.update(my_cfg['backbone'])
    elif my_cfg.__contains__('neck'):
        cfg.model.neck.update(my_cfg['neck'])
    elif my_cfg.__contains__('rpn_head'):
        cfg.model.rpn_head.update(my_cfg['rpn_head'])
    elif my_cfg.__contains__('roi_head'):
        cfg.model.roi_head.update(my_cfg['roi_head'])
    # update data config
    cfg.data.samples_per_gpu = my_cfg['runtime']['batch_size']
    cfg.data_root = my_cfg['data']['data_path']
    cfg.data.train.img_prefix = os.path.join(cfg.data_root, 'train2017')
    cfg.data.train.ann_file = os.path.join(
        cfg.data_root, 'annotations/instances_train2017.json')
    # val data
    cfg.data.val.img_prefix = os.path.join(cfg.data_root, 'val2017')
    cfg.data.val.ann_file = os.path.join(cfg.data_root,
                                         'annotations/instances_val2017.json')
    # test data
    cfg.data.test.img_prefix = os.path.join(cfg.data_root, 'val2017')
    cfg.data.test.ann_file = os.path.join(
        cfg.data_root, 'annotations/instances_val2017.json')
    # update optimizer
    if my_cfg.__contains__('optimizer'):
        cfg.optimizer.update(my_cfg['optimizer'])
    # update runtime
    cfg.total_epochs = my_cfg['runtime']['total_epochs']
    cfg.work_dir = my_cfg['runtime']['work_dir']
    print(cfg)
    print('-------------------')
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
