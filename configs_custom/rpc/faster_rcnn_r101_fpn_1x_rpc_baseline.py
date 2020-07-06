# model settings
model = dict(
    type='DPS_Detector',
    pretrained='torchvision://resnet101',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='SharedFCBBoxHead',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=201,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    
    ### heatmap/density map head
    density_head=dict(
        type='DensityHead',
        output_layer="relu",
        loss="mse",
        in_channels=256,
        num_classes=1,
        loss_weight=1.0), 
        )

# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=100)
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
)
# dataset settings
dataset_type = 'RPC_Dataset'
data_root = '/data/RPC_dataset/RPC_dataset/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# This step is determined by pre-train model's data, not training data!
# test set (119.52966185149884, 125.33286295874348, 129.53535471372086)
# train set (125.32833173447472, 130.372177438396, 132.58775202118858)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations_with_density', with_bbox=True, with_density=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='CustomizedFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', "gt_density_map"]),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=4,
    workers_per_gpu=4,
    select_ratio=0.2,
    train=[
        dict(
        type='RPC_Syn_Dataset',
        ann_file=data_root + 'synthesize_density_map_0_45_threshold_100k.json',
        img_prefix=data_root + 'synthesize_density_map_0_45_threshold_100k/',
        rendered=False,
        n_class_density_map=1,
        csp=False,
        pipeline=train_pipeline),
        
        dict(
        type='RPC_Syn_Dataset',
        ann_file=data_root + 'synthesize_density_map_0_45_threshold_100k.json',
        img_prefix=data_root + 'synthesize_density_map_0_45_threshold_100k/render_v2/',
        rendered=True, 
        n_class_density_map=1,
        csp=False,
        pipeline=train_pipeline),        
          ],
    val=dict(
        type='RPC_Dataset',
        ann_file=data_root + 'instances_val2019.json',
        img_prefix=data_root + 'val2019/',
        export_result_dir = "./work_dir/faster_rcnn_r101_fpn_mmdet_rpc_baseline/val_result/",
        generate_pseudo_label=False,
        pipeline=test_pipeline),
    test=dict(
        type='RPC_Dataset',
        ann_file=data_root + 'instances_test2019.json',
        img_prefix=data_root + 'test2019/',
        export_result_dir = "./work_dir/faster_rcnn_r101_fpn_mmdet_rpc_baseline/test_result/",
        generate_pseudo_label=True,
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='bbox')
evaluation_test = dict(interval=5, metric='bbox')
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[24, 32])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 36
train_mode = 'baseline'
validate = True
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dir/faster_rcnn_r101_fpn_mmdet_rpc_baseline/'
load_from = None
resume_from = './work_dir/faster_rcnn_r101_fpn_mmdet_rpc_baseline/epoch_18.pth'
workflow = [('train', 1)]