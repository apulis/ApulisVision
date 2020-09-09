import argparse

from mmcv import Config

backbone_list = ['resnet', 'resnext', 'seresnet', 'serenext']
gap_list = ['GlobalAveragePooling']
opt_list = ['SGD', 'Adam']


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args


def update_configs(input_cfg):
    my_config = dict()
    input_cfg = input_cfg['nodes']
    for mdict in input_cfg:
        # input module
        name = mdict.id.lower()
        if name == 'input':
            if mdict.name.lower() == 'mnist':
                my_config['dataset_type'] = 'MNIST'
            my_config['data'] = mdict.config
        # backbone
        if name == 'backbone':
            if mdict.name.lower() in backbone_list:
                my_config['backbone'] = {}
                my_config['backbone']['type'] = mdict.name.lower()
            my_config['backbone'].update(mdict.config)
        # neck
        if name == 'neck':
            if mdict.name in gap_list:
                my_config['neck'] = {}
                my_config['neck']['type'] = mdict.name.lower()
                my_config['neck'].update(mdict.config)
        # optimizer
        if name == 'optimizer':
            if mdict.name in opt_list:
                my_config['optimizer'] = {}
                my_config['optimizer']['type'] = mdict.name.lower()
                my_config['optimizer'].update(mdict.config)
        # output
        if name == 'output':
            my_config['runtime'] = {}
            my_config['runtime'].update(mdict.config)

    return my_config


def main():
    args = parse_args()
    input_cfg = Config.fromfile(args.config)
    my_config = update_configs(input_cfg)
    print(my_config)


if __name__ == '__main__':
    main()
