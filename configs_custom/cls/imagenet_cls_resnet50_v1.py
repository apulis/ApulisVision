# model settings
model = dict(
    type='Classifier',
    pretrained='torchvision://resnet50',
    num_classes=257,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=0,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'))
# model training and testing settings
train_cfg = None
test_cfg = None    
# dataset settings
dataset_type = 'ImageFolderDataset'
root = '/data/cls_datasets/256_obj/'
input_size = 224
train_pipeline = [
    dict(type='ResizeImage', size= 256),
    dict(type='RandomSizeAndCrop', crop_nopad=False, size=input_size, scale_min=0.66, scale_max=1.5),
    dict(type='RandomHorizontallyFlip'),
    dict(type='RandomVerticalFlip'),
    dict(type='ColorJitter', brightness=0.2, contrast=0.1, saturation=0.2, hue=0.1),
    dict(type='PILToTensor'),
    dict(type='TorchNormalize', mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
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
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 3,
    step=[3, 5])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=100,  # 100 steps and show loss
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 6
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dirs = "./work_dirs/cls/resnet50/"
load_from = None
resume_from = None
workflow = [('train', 1)]