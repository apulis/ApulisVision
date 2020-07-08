# model settings
model = dict(
    type='Classifier',
    pretrained='open-mmlab://res2net101_v1d_26w_4s',
    num_classes=257,
    backbone=dict(
        type='Res2Net',
        depth=101,
        scales=4,
        base_width=26))
# dataset settings
dataset_type = 'MulticlassDataset'
data_root = '/data/cls_datasets/256_obj/'
#data_root = '/home/robin/datasets/hymenoptera_data/'
data = dict(
    samples_per_gpu=32,    
    workers_per_gpu=4, 
    train=dict(
        type=dataset_type,
        ann_file=None,
        img_prefix=data_root + 'train/',
        train_mode = True),
    val=dict(
        type=dataset_type,
        ann_file=None,
        img_prefix=data_root + 'val/',
        train_mode=False),
    test=dict(
        type=dataset_type,
        ann_file=None,
        img_prefix=data_root + 'val/',
        train_mode=False))
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 3,
    step=[12, 18])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,  # 100 steps and show loss
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 1
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dirs = "./work_dirs/cls/resnet50"
load_from = None
resume_from = None
workflow = [('train', 1)]