<<<<<<< HEAD
<<<<<<< HEAD
checkpoint_config = dict(interval=1)
=======
>>>>>>> mmseg/master
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
<<<<<<< HEAD
        dict(type='TextLoggerHook'),
=======
        dict(type='TextLoggerHook', by_epoch=False),
>>>>>>> mmseg/master
=======
# checkpoint saving
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
>>>>>>> mmcls/master
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
<<<<<<< HEAD
<<<<<<< HEAD
=======
cudnn_benchmark = True
>>>>>>> mmseg/master
=======
>>>>>>> mmcls/master
