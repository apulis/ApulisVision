# model settings
norm_cfg = dict(type='SyncBN')
input_size = 513
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
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
ignore_label = 255
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            imglist_name=data_root + 'VOC2012/ImageSets/Segmentation/train.txt',
            img_prefix=data_root + 'VOC2012/',
            seg_prefix=data_root + 'VOC2012/',
        ),
        transforms=[
            dict(type='RandomScale', min_scale=0.5, max_scale=2.0, mode='bilinear'),
            dict(type='RandomCrop', height=513, width=513, image_value=img_norm_cfg['mean'], mask_value=ignore_label),
            dict(type='RandomRotate', p=0.5, degrees=10, mode='bilinear', border_mode='constant', image_value=img_norm_cfg['mean'], mask_value=ignore_label),
            dict(type='GaussianBlur', p=0.5, ksize=7),
            dict(type='HorizontalFlip', p=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ToTensor'),
        ],
    ),
    val=dict(
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            imglist_name=data_root + 'VOC2012/ImageSets/Segmentation/val.txt',
            img_prefix=data_root + 'VOC2012/',
            seg_prefix=data_root + 'VOC2012/',
        ),
        transforms=[
            dict(type='PadIfNeeded', height=513, width=513, image_value=img_norm_cfg['mean'], mask_value=ignore_label),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ToTensor'),
        ],
    ),
    test=dict(
        type=dataset_type,
        imglist_name=data_root + 'VOC2012/ImageSets/Segmentation/val.txt',
        img_prefix=data_root + 'VOC2012/',
        seg_prefix=data_root + 'VOC2012/',
        transforms=[
            dict(type='PadIfNeeded', height=513, width=513, image_value=img_norm_cfg['mean'], mask_value=ignore_label),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ToTensor'),
        ],
    ),
)
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
