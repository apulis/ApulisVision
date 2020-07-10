# model settings
norm_cfg = dict(type='SyncBN')
input_size = 768
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://msra/hrnetv2_w32',
    backbone=dict(
        type='HRNet',
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4,),
                num_channels=(64,)),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))),
        concat_feats=True),
    mask_head=dict(
        type='OCRHead',
        in_channel=480,
        in_channel_aux=480,
        inter_channel_aux=480,
        drop_out=0.05,
        num_classes=7,
        upsample_cfg=dict(type='bilinear', upsample_size=input_size),
        loss_mask=dict(
            type='FocalSegLoss', gamma=0.3, loss_weight=0.527),
        loss_mask_aux=dict(
            type='LovaszSoftmax', loss_weight=0.263),
        loss_context=dict(
            type='CrossEntropySegLoss', loss_weight=0.21)))
# model training and testing settings
train_cfg = None
test_cfg = None
# dataset settings
dataset_type = 'DGLandcoverDataset'
data_root = '/data/landcover/'
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
    dict(type='ResizeImg', size=2432),
    dict(
        type='MultiTestAug',
        aug_list=[
            # dict(type='Resize', size=1920),
            # dict(type='Rotate', degree=90),
            # dict(type='Rotate', degree=270),
            # dict(type='HorizontallyFlip'),
            # dict(type='VerticalFlip')
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
    dict(type='ResizeImg', size=2432),
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
    samples_per_gpu=2,
    workers_per_gpu=2,
    sampler='group',
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'DGland_annotations/DGland_train.txt',
        img_prefix=data_root + 'DGland_train/images/',
        seg_prefix=data_root + 'DGland_train/labels/',
        tile_size=1500,
        stride=1500,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'DGland_annotations/DGland_val.txt',
        img_prefix=data_root + 'DGland_val/images/',
        seg_prefix=data_root + 'DGland_val/labels/',
        tile_size=input_size,
        stride=input_size,
        pipeline=validation_pipeline),
    eval=dict(
        type=dataset_type,
        ann_file=data_root + 'DGland_annotations/DGland_val.txt',
        img_prefix=data_root + 'DGland_val/images/',
        seg_prefix=data_root + 'DGland_val/labels/',
        tile_size=2448,
        stride=2448,
        pipeline=eval_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'DGland_annotations/DGland_val.txt',
        data_path=data_root + 'DGland_validation/images/',
        tile_size=2448,
        stride=2448,
        test_mode=True,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.006, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    power = 1.5)
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=2,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
evaluation = dict(interval=1)
# runtime settings
total_epochs = 120
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dirs = './work_dirs/ocrnet_hrnet32_fl+lvsl_DGLandcover_1'
label_suffix = '_mask'
load_from = None
resume_from = None
workflow = [('train', 5), ('val', 1)]
