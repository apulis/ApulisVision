# model settings
norm_cfg = dict(type='SyncBN')
input_size = 768
model = dict(
    type='EncoderDecoder',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNetV1s',
        depth=50,
        in_channels=3,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        frozen_stages=-1,
        deep_stem=False,
        style='pytorch'),
    neck=dict(
        type='Decoder',
        in_channels=[256, 512, 1024, 2048],
        inter_channels=[1024, 512, 256, 256],
        center_channels=2048,
        out_channel=128),
    mask_head=dict(
        type='FCNHead',
        upsample_size=input_size,
        num_convs=1,
        in_channels=128,
        conv_out_channels=128,
        drop_out=0.1,
        upsample_method=None,
        num_classes=7,
        loss_mask=dict(
            type='FocalLoss', gamma=0.3, loss_weight=1.0),
        loss_mask_aux=dict(
            type='LovaszSoftmax', loss_weight=0.5)))
# model training and testing settings
train_cfg = None
test_cfg = None
# dataset settings
dataset_type = 'DGLandcoverDataset'
img_norm_cfg = dict(
    normalize=True, mean=[104.0936, 96.6689, 71.8065], std=[37.4711, 29.2703, 26.8835])

train_pipeline = [
    dict(type='ReadImage'),
    dict(type='ToPilImage'),
    dict(type='RandomSizeAndCrop', size=input_size, scale_min=0.66, scale_max=1.5),
    dict(type='RandomHorizontallyFlip'),
    dict(type='RandomVerticalFlip'),
    dict(type='RandomGaussianBlur'),
    dict(type='ColorJitter', brightness=0.2, contrast=0.1, saturation=0.2, hue=0.1),
    dict(type='ToArray', **img_norm_cfg),
    #dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_semantic_seg'],
        meta_keys=('img_id', 'ori_shape', 'img_shape', 'img_norm_cfg', 'horizontal_flip', 'vertical_flip', 'h', 'w'))
]
validation_pipeline = [
    dict(type='ReadImage'),
    dict(type='ToPilImage'),
    dict(type='ToArray', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img', 'gt_semantic_seg'],
        meta_keys=('img_id', 'ori_shape', 'img_shape', 'img_norm_cfg', 'h', 'w'))]
eval_pipeline = [
    dict(type='ReadImage'),
    dict(type='ToPilImage'),
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
                meta_keys=('img_id', 'ori_shape', 'horizontal_flip', 'vertical_flip', 'rotate', 'full_shape', 'tile_size', 'img_shape', 'img_norm_cfg', 'h', 'w'))])]
test_pipeline = [
    dict(type='ReadImage'),
    dict(type='ToPilImage'),
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
                meta_keys=('img_id', 'ori_shape', 'horizontal_flip', 'vertical_flip', 'rotate', 'full_shape', 'tile_size', 'img_shape', 'img_norm_cfg', 'h', 'w'))])]
test_aug_weights = None

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        pipeline=train_pipeline,
        data_path='/data/landcover/DGland_train/images',
        label_path='/data/landcover/DGland_train/labels',
        tile_size=1500,
        stride=1500,
        mode='train'),
    val=dict(
        type=dataset_type,
        pipeline=validation_pipeline,
        data_path='/data/landcover/DGland_val/images',
        label_path='/data/landcover/DGland_val/labels',
        tile_size=input_size,
        stride=input_size,
        mode='val'),
    eval=dict(
        type=dataset_type,
        pipeline=eval_pipeline,
        data_path='/data/landcover/DGland_val/images',
        label_path='/data/landcover/DGland_val/labels',
        tile_size=1280,
        stride=1280,
        mode='val'),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        data_path='/data/landcover/DGland_validation/images',
        tile_size=1280,
        stride=1280,
        mode='test'))
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
#optimizer_config = None
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    gamma=0.63,
    step=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 70, 80, 90])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
evaluation = dict(interval=1)
# runtime settings
total_epochs = 150
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dir/unet_r50_fl+lvsl_DGLandcover_1'
label_suffix = '_mask'
load_from = None
resume_from = None
workflow = [('train', 5), ('val', 1)]