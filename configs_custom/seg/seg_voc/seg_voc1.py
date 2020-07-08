# model settings
norm_cfg = dict(type='SyncBN')
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
        num_classes=7,
        loss_mask=dict(
            type='CrossEntropyLoss', loss_weight=1.0),
        loss_mask_aux=dict(
            type='LovaszSoftmax', loss_weight=0.5)))
# model training and testing settings
train_cfg = None
test_cfg = None
# dataset settings
dataset_type = 'VOCSegmentation'
data_root = '/data/voc/VOCdevkit/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=False, with_label=False, with_mask=False, with_seg=True),
    dict(type='Resize', img_scale=(600, 600), keep_ratio=False),
    dict(type='RandomCrop', crop_size=(input_size, input_size)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=False, with_label=False, with_mask=False, with_seg=True),
    dict(type='Resize', img_scale=(600, 600), keep_ratio=False),
    dict(type='RandomCrop', crop_size=(input_size, input_size)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'VOC2012/ImageSets/Segmentation/train.txt',
        img_prefix=data_root + 'VOC2012/JPEGImages/',
        seg_prefix=data_root + 'VOC2012/SegmentationClass/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2012/ImageSets/Segmentation/val.txt',
        img_prefix=data_root + 'VOC2012/JPEGImages/',
        seg_prefix=data_root + 'VOC2012/SegmentationClass/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2012/ImageSets/Segmentation/val.txt',
        img_prefix=data_root + 'VOC2012/JPEGImages/',
        seg_prefix=data_root + 'VOC2012/SegmentationClass/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[3])  # actual epoch = 3 * 3 = 9
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 4  # actual epoch = 4 * 3 = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dirs = './work_dirs/deeplabv3_r50_dlt_cel'
load_from = None
resume_from = None
workflow = [('train', 1)]
