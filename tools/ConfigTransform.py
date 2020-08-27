from mmcv import Config
import os


# json转换为config.py文件
class JsonTransformer():
    def __init__(self):
        self.jobId = 1

    def toConfig(self, models,type, oriConfig):
        '''
        @param mergeDict: 需要merge的字典
        @param oriConfig: 原始的config文件路径
        @return:
        '''
        self.models =models
        self.type = type
        self.configFile = oriConfig
        if (self.type == 'cls'):
            return self.gen_cls_config()
        elif (self.type == 'det'):
            return self.gen_cls_config()
    def gen_input(self):
        input = self.models[0]['config']
        dataset_type = 'ImageFolderDataset'
        if 'objectcategories' in input['name'].lower():
            root = '/data/cls_datasets/256_obj/'

        #     TODO:test
        root = '/data/cls_datasets/256_obj/'
        input_size = 224

        # 测试使用小数据集合
        train_pipeline = [
            dict(type='ResizeImage', size=256),
            dict(type='RandomSizeAndCrop', crop_nopad=False, size=input_size, scale_min=0.66, scale_max=1.5),
            dict(type='RandomHorizontallyFlip'),
            dict(type='RandomVerticalFlip'),
            dict(type='ColorJitter', brightness=0.2, contrast=0.1, saturation=0.2, hue=0.1),
            dict(type='PILToTensor'),
            dict(type='TorchNormalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            dict(type='Collect', keys=['img', 'gt_labels'], meta_keys=['filename'])
        ]
        data = dict(
            samples_per_gpu=32,
            workers_per_gpu=4,
            train=dict(
                type=dataset_type,
                data_root=root + 'train/',
                pipeline=train_pipeline),
            val=dict(
                type=dataset_type,
                data_root=root + 'val/',
                pipeline=train_pipeline),
            test=dict(
                type=dataset_type,
                data_root=root + 'val/',
                pipeline=train_pipeline),
        )
        return dict(data=data)

    def gen_model(self):
        '''

        @return:返回model配置文件
        '''
        #     TODO:test
        num_classes = 257
        backbone = self.models[1]['config']
        polling = self.models[2]['config']
        if 'resnet' in backbone['name'].lower():
            depth = int(backbone['depth'])
        if 'max' in polling['name'].lower():
            global_pool = 'max'
        else:
            global_pool = 'avg'
        model = dict(backbone=dict(depth=depth), num_classes=num_classes, global_pool=global_pool)
        return dict(model=model)

    def gen_optimizer(self):
        optimizer = self.models[3]['config']
        if 'sgd' in optimizer['name'].lower():
            optimizer_type = 'SGD'
        else:
            optimizer_type = 'ADAM'
        print(optimizer)
        lr = float(optimizer['learning_rate'])
        optimizer = dict(type=optimizer_type, lr=lr, momentum=0.9, weight_decay=0.0001)
        return dict(optimizer=optimizer)

    def gen_output(self):
        output = self.models[4]['config']
        total_epochs = int(output['total_epochs'])
        warmup_ratio = float(output['warmup_ratio'])
        warmup_iters = float(output['warmup_iters'])
        self.work_dir = output['work_dir']

        lr_config = dict(
            policy='step',
            warmup='linear',
            warmup_iters=warmup_iters,
            warmup_ratio=warmup_ratio,
            step=[12, 18])
        resume_from = None
        log_config = dict(
            interval=2,  # 100 steps and show loss
            hooks=[
                dict(type='TextLoggerHook'),
                dict(type='TensorboardLoggerHook')
            ])
        output = dict(total_epochs=total_epochs, lr_config=lr_config, work_dir=self.work_dir, resume_from=resume_from,
                      log_config=log_config)
        return output

    def gen_cls_config(self):
        # 合并字典
        config_dict = {**self.gen_input(), **self.gen_model(), **self.gen_optimizer(), **self.gen_output()}
        # 生成config
        config = Config.fromfile(self.configFile)
        config.merge_from_dict(config_dict)
        return config

    def toJson(self, configFile):
        return Config.fromfile(configFile).__getattribute__('_cfg_dict')

