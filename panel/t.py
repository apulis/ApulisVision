model = dict(
    type='FasterRCNN',
    pretrained='torchvision://resnet50',
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8]),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
        ),
    ))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    step=[8, 11])
total_epochs = 12

