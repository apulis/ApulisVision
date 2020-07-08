# model settings
norm_cfg = dict(type='SyncBN')
num_classes = 150
input_size = 512
model = dict(
    type='FCN',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNetV1s',
        depth=50,
        in_channels=3,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        strides=(1, 2, 1, 1),
        dilations=(1, 1, 2, 4),
        deep_stem = False,
        frozen_stages=-1,
        style='pytorch'),
    neck=dict(
        type='ASPP',
        in_channels=2048,
        atrous_rates=[12, 24, 36]),
    mask_head=dict(
        type='FCNHead',
        num_convs=1,
        in_channels=256,
        conv_out_channels=256,
        drop_out=0.1,
        upsample_cfg=dict(type='bilinear', upsample_size=input_size),
        num_classes=num_classes,
        loss_mask=dict(
            type='CrossEntropyLoss', loss_weight=1.0),
        loss_mask_aux=dict(
            type='LovaszSoftmax', loss_weight=0.5)))
# model training and testing settings
train_cfg = None
test_cfg = None
# dataset settings
dataset_type = 'ADE20KSegmentation'
data_root = '/data/semantic_seg/ade20k/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], normalize=True)
train_pipeline = [
    dict(type='ReadPILImage'),
    dict(type='RandomSizeAndCrop', crop_nopad=False, size=input_size, scale_min=0.66, scale_max=1.5),
    dict(type='RandomHorizontallyFlip'),
    dict(type='RandomVerticalFlip'),
    dict(type='RandomGaussianBlur'),
    dict(type='ColorJitter', brightness=0.2, contrast=0.1, saturation=0.2, hue=0.1),
    dict(type='ToArray', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img', 'gt_semantic_seg'],
        meta_keys=('img_id', 'img_norm_cfg', 'horizontal_flip', 'vertical_flip', 'h', 'w'))
]
validation_pipeline = [
    dict(type='ReadPILImage'),
    dict(type='ToArray', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img', 'gt_semantic_seg'],
        meta_keys=('img_id', 'img_shape', 'img_norm_cfg', 'h', 'w'))]
eval_pipeline = [
    dict(type='ReadPILImage'),
    dict(
        type='MultiTestAug',
        aug_list=[
            #dict(type='Resize', size=1920),
            #dict(type='Rotate', degree=90),
            #dict(type='Rotate', degree=270),
            #dict(type='HorizontallyFlip'),
            #dict(type='VerticalFlip')
        ],
        transforms=[
            dict(type='ToArray', **img_norm_cfg),
            dict(
                type='Collect',
                keys=['img', 'gt_semantic_seg'],
                meta_keys=('img_id', 'horizontal_flip', 'vertical_flip', 'rotate', 'full_shape', 'tile_size', 'img_shape', 'img_norm_cfg', 'h', 'w'))])]
test_pipeline = [
    dict(type='ReadPILImage'),
    dict(
        type='MultiTestAug',
        aug_list=[
            dict(type='Resize', size=1280),
            dict(type='Rotate', degree=90),
            dict(type='Rotate', degree=180),
            dict(type='Rotate', degree=270)],
        transforms=[
            dict(type='ToArray', **img_norm_cfg),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=('img_id', 'horizontal_flip', 'vertical_flip', 'rotate', 'full_shape', 'tile_size', 'img_shape', 'img_norm_cfg', 'h', 'w'))])]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        imglist_name=data_root + 'ADEChallengeData2016/train.txt',
        img_prefix=data_root + 'ADEChallengeData2016/images/training/',
        seg_prefix=data_root + 'ADEChallengeData2016/annotations/training/',
        test_mode = False,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        imglist_name=data_root + 'ADEChallengeData2016/val.txt',
        img_prefix=data_root + 'ADEChallengeData2016/images/validation/',
        seg_prefix=data_root + 'ADEChallengeData2016/annotations/validation/',
        test_mode = True,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        imglist_name=data_root + 'ADEChallengeData2016/val.txt',
        img_prefix=data_root + 'ADEChallengeData2016/images/validation/',
        seg_prefix=data_root + 'ADEChallengeData2016/annotations/validation/',
        test_mode = True,
        pipeline=test_pipeline)
)
# optimizer
optimizer = dict(type='SGD', lr=0.007, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    power = 2)
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
evaluation = dict(interval=1)
# runtime settings
total_epochs = 50 
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dirs = './work_dirs/ade20k_deeplabv3_r50_dlt_cel'
load_from = None
resume_from = None
workflow = [('train', 1)]
