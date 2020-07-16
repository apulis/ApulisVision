<<<<<<< HEAD
# High-resolution networks (HRNets) for object detection

## Introduction

=======
# Deep High-Resolution Representation Learning for Human Pose Estimation

## Introduction
>>>>>>> mmseg/master
```
@inproceedings{SunXLW19,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Ke Sun and Bin Xiao and Dong Liu and Jingdong Wang},
  booktitle={CVPR},
  year={2019}
}
<<<<<<< HEAD

@article{SunZJCXLMWLW19,
  title={High-Resolution Representations for Labeling Pixels and Regions},
  author={Ke Sun and Yang Zhao and Borui Jiang and Tianheng Cheng and Bin Xiao
  and Dong Liu and Yadong Mu and Xinggang Wang and Wenyu Liu and Jingdong Wang},
  journal   = {CoRR},
  volume    = {abs/1904.04514},
  year={2019}
}
```

## Results and Models


### Faster R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Download |
| :-------------: | :-----: | :-----: | :------: | :-------------:|:------:| :-------:|
|   HRNetV2p-W18  | pytorch |   1x    | 6.6      | 13.4           | 36.9   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w18_1x_coco/faster_rcnn_hrnetv2p_w18_1x_coco_20200130-56651a6d.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w18_1x_coco/faster_rcnn_hrnetv2p_w18_1x_coco_20200130_211246.log.json) |
|   HRNetV2p-W18  | pytorch |   2x    |          |                |        | |
|   HRNetV2p-W32  | pytorch |   1x    | 9.0      | 12.4           | 40.2   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w32_1x_coco/faster_rcnn_hrnetv2p_w32_1x_coco_20200130-6e286425.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w32_1x_coco/faster_rcnn_hrnetv2p_w32_1x_coco_20200130_204442.log.json) |
|   HRNetV2p-W32  | pytorch |   2x    | 9.0        |              | 41.4   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w32_2x_coco/faster_rcnn_hrnetv2p_w32_2x_coco_20200529_015927-976a9c15.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w32_2x_coco/faster_rcnn_hrnetv2p_w32_2x_coco_20200529_015927.log.json)  |
|   HRNetV2p-W40  | pytorch |   1x    | 10.4     | 10.5           | 41.2   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w40_1x_coco/faster_rcnn_hrnetv2p_w40_1x_coco_20200210-95c1f5ce.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w40_1x_coco/faster_rcnn_hrnetv2p_w40_1x_coco_20200210_125315.log.json) |
|   HRNetV2p-W40  | pytorch |   2x    | 10.4     |                |  42.1  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w40_2x_coco/faster_rcnn_hrnetv2p_w40_2x_coco_20200512_161033-0f236ef4.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w40_2x_coco/faster_rcnn_hrnetv2p_w40_2x_coco_20200512_161033.log.json)  |

### Mask R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Download |
| :-------------: | :-----: | :-----: | :------: | :-------------:|:------:| :------:|:--------:|
|   HRNetV2p-W18  | pytorch |   1x    | 7.0      | 11.7           | 37.7   | 34.2    | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w18_1x_coco/mask_rcnn_hrnetv2p_w18_1x_coco_20200205-1c3d78ed.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w18_1x_coco/mask_rcnn_hrnetv2p_w18_1x_coco_20200205_232523.log.json) |
|   HRNetV2p-W18  | pytorch |   2x    | 7.0      | -              | 39.8   | 36.0    | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w18_2x_coco/mask_rcnn_hrnetv2p_w18_2x_coco_20200212-b3c825b1.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w18_2x_coco/mask_rcnn_hrnetv2p_w18_2x_coco_20200212_134222.log.json) |
|   HRNetV2p-W32  | pytorch |   1x    | 9.4      | 11.3           | 41.2   | 37.1    | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w32_1x_coco/mask_rcnn_hrnetv2p_w32_1x_coco_20200207-b29f616e.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w32_1x_coco/mask_rcnn_hrnetv2p_w32_1x_coco_20200207_055017.log.json) |
|   HRNetV2p-W32  | pytorch |   2x    | 9.4      | -              | 42.5   | 37.8    | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w32_2x_coco/mask_rcnn_hrnetv2p_w32_2x_coco_20200213-45b75b4d.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w32_2x_coco/mask_rcnn_hrnetv2p_w32_2x_coco_20200213_150518.log.json) |
|   HRNetV2p-W40  | pytorch |   1x    |  10.9    |                | 42.1   |  37.5   |  [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w40_1x_coco/mask_rcnn_hrnetv2p_w40_1x_coco_20200511_015646-66738b35.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w40_1x_coco/mask_rcnn_hrnetv2p_w40_1x_coco_20200511_015646.log.json)  |
|   HRNetV2p-W40  | pytorch |   2x    |   10.9   |                | 42.8   |  38.2   |  [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w40_2x_coco/mask_rcnn_hrnetv2p_w40_2x_coco_20200512_163732-aed5e4ab.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w40_2x_coco/mask_rcnn_hrnetv2p_w40_2x_coco_20200512_163732.log.json)  |


### Cascade R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Download |
| :-------------: | :-----: | :-----: | :------: | :-------------:|:------:| :-------:|
|   HRNetV2p-W18  | pytorch |   20e   |  7.0     | 11.0           | 41.2   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/cascade_rcnn_hrnetv2p_w18_20e_coco/cascade_rcnn_hrnetv2p_w18_20e_coco_20200210-434be9d7.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/cascade_rcnn_hrnetv2p_w18_20e_coco/cascade_rcnn_hrnetv2p_w18_20e_coco_20200210_105632.log.json)  |
|   HRNetV2p-W32  | pytorch |   20e   |  9.4     | 11.0           | 43.3   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/cascade_rcnn_hrnetv2p_w32_20e_coco/cascade_rcnn_hrnetv2p_w32_20e_coco_20200208-928455a4.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/cascade_rcnn_hrnetv2p_w32_20e_coco/cascade_rcnn_hrnetv2p_w32_20e_coco_20200208_160511.log.json)  |
|   HRNetV2p-W40  | pytorch |   20e   |  10.8    |                | 43.8   |  [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/cascade_rcnn_hrnetv2p_w40_20e_coco/cascade_rcnn_hrnetv2p_w40_20e_coco_20200512_161112-75e47b04.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/cascade_rcnn_hrnetv2p_w40_20e_coco/cascade_rcnn_hrnetv2p_w40_20e_coco_20200512_161112.log.json)  |


### Cascade Mask R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Download |
| :-------------: | :-----: | :-----: | :------: | :-------------:|:------:| :------:|:--------:|
|   HRNetV2p-W18  | pytorch |   20e   | 8.5      | 8.5            |41.6    |36.4     |  [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/cascade_mask_rcnn_hrnetv2p_w18_20e_coco/cascade_mask_rcnn_hrnetv2p_w18_20e_coco_20200210-b543cd2b.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/cascade_mask_rcnn_hrnetv2p_w18_20e_coco/cascade_mask_rcnn_hrnetv2p_w18_20e_coco_20200210_093149.log.json)  |
|   HRNetV2p-W32  | pytorch |   20e   |          | 8.3            |44.3    |38.6     |  [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/cascade_mask_rcnn_hrnetv2p_w32_20e_coco/cascade_mask_rcnn_hrnetv2p_w32_20e_coco_20200512_154043-39d9cf7b.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/cascade_mask_rcnn_hrnetv2p_w32_20e_coco/cascade_mask_rcnn_hrnetv2p_w32_20e_coco_20200512_154043.log.json)  |
|   HRNetV2p-W40  | pytorch |   20e   | 12.5     |                |45.1    |39.3     |  [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/cascade_mask_rcnn_hrnetv2p_w40_20e_coco/cascade_mask_rcnn_hrnetv2p_w40_20e_coco_20200527_204922-969c4610.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/cascade_mask_rcnn_hrnetv2p_w40_20e_coco/cascade_mask_rcnn_hrnetv2p_w40_20e_coco_20200527_204922.log.json)    |

### Hybrid Task Cascade (HTC)

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Download |
| :-------------: | :-----: | :-----: | :------: | :-------------:|:------:| :------:|:--------:|
|   HRNetV2p-W18  | pytorch |   20e   | 10.8     | 4.7            | 42.8   | 37.9    | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/htc_hrnetv2p_w18_20e_coco/htc_hrnetv2p_w18_20e_coco_20200210-b266988c.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/htc_hrnetv2p_w18_20e_coco/htc_hrnetv2p_w18_20e_coco_20200210_182735.log.json) |
|   HRNetV2p-W32  | pytorch |   20e   | 13.1     | 4.9            | 45.4   | 39.9    | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/htc_hrnetv2p_w32_20e_coco/htc_hrnetv2p_w32_20e_coco_20200207-7639fa12.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/htc_hrnetv2p_w32_20e_coco/htc_hrnetv2p_w32_20e_coco_20200207_193153.log.json) |
|   HRNetV2p-W40  | pytorch |   20e   | 14.6     |                | 46.4   | 40.8    | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/htc_hrnetv2p_w40_20e_coco/htc_hrnetv2p_w40_20e_coco_20200529_183411-417c4d5b.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/htc_hrnetv2p_w40_20e_coco/htc_hrnetv2p_w40_20e_coco_20200529_183411.log.json) |


### FCOS

| Backbone  | Style   |  GN     | MS train | Lr schd | Mem (GB) | Inf time (fps) | box AP | Download |
|:---------:|:-------:|:-------:|:--------:|:-------:|:------:|:------:|:------:|:--------:|
|HRNetV2p-W18| pytorch | Y       | N       | 1x       | 13.0 | 12.9 | 35.1   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w18_gn-head_4x4_1x_coco/fcos_hrnetv2p_w18_gn-head_4x4_1x_coco_20200316-c24bac34.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w18_gn-head_4x4_1x_coco/fcos_hrnetv2p_w18_gn-head_4x4_1x_coco_20200316_103815.log.json) |
|HRNetV2p-W18| pytorch | Y       | N       | 2x       | 13.0 | -    | 37.7   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w18_gn-head_4x4_2x_coco/fcos_hrnetv2p_w18_gn-head_4x4_2x_coco_20200316-15348c5b.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w18_gn-head_4x4_2x_coco/fcos_hrnetv2p_w18_gn-head_4x4_2x_coco_20200316_103815.log.json) |
|HRNetV2p-W32| pytorch | Y       | N       | 1x       | 17.5 | 12.9 | 39.2   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w32_gn-head_4x4_1x_coco/fcos_hrnetv2p_w32_gn-head_4x4_1x_coco_20200314-59a7807f.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w32_gn-head_4x4_1x_coco/fcos_hrnetv2p_w32_gn-head_4x4_1x_coco_20200314_150555.log.json) |
|HRNetV2p-W32| pytorch | Y       | N       | 2x       | 17.5 | -    | 40.3   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w32_gn-head_4x4_2x_coco/fcos_hrnetv2p_w32_gn-head_4x4_2x_coco_20200314-faf8f0b8.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w32_gn-head_4x4_2x_coco/fcos_hrnetv2p_w32_gn-head_4x4_2x_coco_20200314_145136.log.json) |
|HRNetV2p-W18| pytorch | Y       | Y       | 2x       | 13.0 | 12.9 | 38.1   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w18_gn-head_mstrain_640-800_4x4_2x_coco/fcos_hrnetv2p_w18_gn-head_mstrain_640-800_4x4_2x_coco_20200316-a668468b.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w18_gn-head_mstrain_640-800_4x4_2x_coco/fcos_hrnetv2p_w18_gn-head_mstrain_640-800_4x4_2x_coco_20200316_104027.log.json) |
|HRNetV2p-W32| pytorch | Y       | Y       | 2x       | 17.5 | 12.4 | 41.8   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w32_gn-head_mstrain_640-800_4x4_2x_coco/fcos_hrnetv2p_w32_gn-head_mstrain_640-800_4x4_2x_coco_20200314-065d37a6.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w32_gn-head_mstrain_640-800_4x4_2x_coco/fcos_hrnetv2p_w32_gn-head_mstrain_640-800_4x4_2x_coco_20200314_145356.log.json) |
|HRNetV2p-W48| pytorch | Y       | Y       | 2x       | 20.3 | 10.8 | 42.8   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w40_gn-head_mstrain_640-800_4x4_2x_coco/fcos_hrnetv2p_w40_gn-head_mstrain_640-800_4x4_2x_coco_20200314-e201886d.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w40_gn-head_mstrain_640-800_4x4_2x_coco/fcos_hrnetv2p_w40_gn-head_mstrain_640-800_4x4_2x_coco_20200314_150607.log.json) |



**Note:**

- The `28e` schedule in HTC indicates decreasing the lr at 24 and 27 epochs, with a total of 28 epochs.
- HRNetV2 ImageNet pretrained models are in [HRNets for Image Classification](https://github.com/HRNet/HRNet-Image-Classification).
=======
```

## Results and models

### Cityscapes
| Method |      Backbone      | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) |                                                                                                                                                                                        download                                                                                                                                                                                        |
|--------|--------------------|-----------|--------:|----------|----------------|------:|--------------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| FCN    | HRNetV2p-W18-Small | 512x1024  |   40000 |      1.7 |          23.74 | 73.86 |         75.91 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr18s_512x1024_40k_cityscapes/fcn_hr18s_512x1024_40k_cityscapes_20200601_014216-93db27d0.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr18s_512x1024_40k_cityscapes/fcn_hr18s_512x1024_40k_cityscapes_20200601_014216.log.json)     |
| FCN    | HRNetV2p-W18       | 512x1024  |   40000 |      2.9 |          12.97 | 77.19 |         78.92 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr18_512x1024_40k_cityscapes/fcn_hr18_512x1024_40k_cityscapes_20200601_014216-f196fb4e.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr18_512x1024_40k_cityscapes/fcn_hr18_512x1024_40k_cityscapes_20200601_014216.log.json)         |
| FCN    | HRNetV2p-W48       | 512x1024  |   40000 |      6.2 |           6.42 | 78.48 |         79.69 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr48_512x1024_40k_cityscapes/fcn_hr48_512x1024_40k_cityscapes_20200601_014240-a989b146.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr48_512x1024_40k_cityscapes/fcn_hr48_512x1024_40k_cityscapes_20200601_014240.log.json)         |
| FCN    | HRNetV2p-W18-Small | 512x1024  |   80000 | -        | -              | 75.31 |         77.48 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr18s_512x1024_80k_cityscapes/fcn_hr18s_512x1024_80k_cityscapes_20200601_202700-1462b75d.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr18s_512x1024_80k_cityscapes/fcn_hr18s_512x1024_80k_cityscapes_20200601_202700.log.json)     |
| FCN    | HRNetV2p-W18       | 512x1024  |   80000 | -        | -              | 78.65 |         80.35 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr18_512x1024_80k_cityscapes/fcn_hr18_512x1024_80k_cityscapes_20200601_223255-4e7b345e.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr18_512x1024_80k_cityscapes/fcn_hr18_512x1024_80k_cityscapes_20200601_223255.log.json)         |
| FCN    | HRNetV2p-W48       | 512x1024  |   80000 | -        | -              | 79.93 |         80.72 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr48_512x1024_80k_cityscapes/fcn_hr48_512x1024_80k_cityscapes_20200601_202606-58ea95d6.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr48_512x1024_80k_cityscapes/fcn_hr48_512x1024_80k_cityscapes_20200601_202606.log.json)         |
| FCN    | HRNetV2p-W18-Small | 512x1024  |  160000 | -        | -              | 76.31 |         78.31 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr18s_512x1024_160k_cityscapes/fcn_hr18s_512x1024_160k_cityscapes_20200602_190901-4a0797ea.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr18s_512x1024_160k_cityscapes/fcn_hr18s_512x1024_160k_cityscapes_20200602_190901.log.json) |
| FCN    | HRNetV2p-W18       | 512x1024  |  160000 | -        | -              | 78.80 |         80.74 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr18_512x1024_160k_cityscapes/fcn_hr18_512x1024_160k_cityscapes_20200602_190822-221e4a4f.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr18_512x1024_160k_cityscapes/fcn_hr18_512x1024_160k_cityscapes_20200602_190822.log.json)     |
| FCN    | HRNetV2p-W48       | 512x1024  |  160000 | -        | -              | 80.65 |         81.92 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr48_512x1024_160k_cityscapes/fcn_hr48_512x1024_160k_cityscapes_20200602_190946-59b7973e.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr48_512x1024_160k_cityscapes/fcn_hr48_512x1024_160k_cityscapes_20200602_190946.log.json)     |

### ADE20K
| Method |      Backbone      | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) |                                                                                                                                                                              download                                                                                                                                                                              |
|--------|--------------------|-----------|--------:|----------|----------------|------:|--------------:|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| FCN    | HRNetV2p-W18-Small | 512x512   |   80000 |      3.8 |          38.66 | 31.38 |         32.45 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr18s_512x512_80k_ade20k/fcn_hr18s_512x512_80k_ade20k_20200614_144345-77fc814a.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr18s_512x512_80k_ade20k/fcn_hr18s_512x512_80k_ade20k_20200614_144345.log.json)     |
| FCN    | HRNetV2p-W18       | 512x512   |   80000 |      4.9 |          22.57 | 35.51 |         36.80 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr18_512x512_80k_ade20k/fcn_hr18_512x512_80k_ade20k_20200614_185145-66f20cb7.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr18_512x512_80k_ade20k/fcn_hr18_512x512_80k_ade20k_20200614_185145.log.json)         |
| FCN    | HRNetV2p-W48       | 512x512   |   80000 |      8.2 |          21.23 | 41.90 |         43.27 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr48_512x512_80k_ade20k/fcn_hr48_512x512_80k_ade20k_20200614_193946-7ba5258d.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr48_512x512_80k_ade20k/fcn_hr48_512x512_80k_ade20k_20200614_193946.log.json)         |
| FCN    | HRNetV2p-W18-Small | 512x512   |  160000 | -        | -              | 33.00 |         34.55 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr18s_512x512_160k_ade20k/fcn_hr18s_512x512_160k_ade20k_20200614_214413-870f65ac.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr18s_512x512_160k_ade20k/fcn_hr18s_512x512_160k_ade20k_20200614_214413.log.json) |
| FCN    | HRNetV2p-W18       | 512x512   |  160000 | -        | -              | 36.79 |         38.58 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr18_512x512_160k_ade20k/fcn_hr18_512x512_160k_ade20k_20200614_214426-ca961836.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr18_512x512_160k_ade20k/fcn_hr18_512x512_160k_ade20k_20200614_214426.log.json)     |
| FCN    | HRNetV2p-W48       | 512x512   |  160000 | -        | -              | 42.02 |         43.86 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr48_512x512_160k_ade20k/fcn_hr48_512x512_160k_ade20k_20200614_214407-a52fc02c.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr48_512x512_160k_ade20k/fcn_hr48_512x512_160k_ade20k_20200614_214407.log.json)     |

### Pascal VOC 2012 + Aug
| Method |      Backbone      | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) |                                                                                                                                                                                download                                                                                                                                                                                |
|--------|--------------------|-----------|--------:|----------|----------------|------:|--------------:|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| FCN    | HRNetV2p-W18-Small | 512x512   |   20000 |      1.8 |          43.36 | 65.20 |         68.55 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr18s_512x512_20k_voc12aug/fcn_hr18s_512x512_20k_voc12aug_20200617_224503-56e36088.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr18s_512x512_20k_voc12aug/fcn_hr18s_512x512_20k_voc12aug_20200617_224503.log.json) |
| FCN    | HRNetV2p-W18       | 512x512   |   20000 |      2.9 |          23.48 | 72.30 |         74.71 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr18_512x512_20k_voc12aug/fcn_hr18_512x512_20k_voc12aug_20200617_224503-488d45f7.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr18_512x512_20k_voc12aug/fcn_hr18_512x512_20k_voc12aug_20200617_224503.log.json)     |
| FCN    | HRNetV2p-W48       | 512x512   |   20000 |      6.2 |          22.05 | 75.87 |         78.58 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr48_512x512_20k_voc12aug/fcn_hr48_512x512_20k_voc12aug_20200617_224419-89de05cd.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr48_512x512_20k_voc12aug/fcn_hr48_512x512_20k_voc12aug_20200617_224419.log.json)     |
| FCN    | HRNetV2p-W18-Small | 512x512   |   40000 | -        | -              | 66.61 |         70.00 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr18s_512x512_40k_voc12aug/fcn_hr18s_512x512_40k_voc12aug_20200614_000648-4f8d6e7f.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr18s_512x512_40k_voc12aug/fcn_hr18s_512x512_40k_voc12aug_20200614_000648.log.json) |
| FCN    | HRNetV2p-W18       | 512x512   |   40000 | -        | -              | 72.90 |         75.59 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr18_512x512_40k_voc12aug/fcn_hr18_512x512_40k_voc12aug_20200613_224401-1b4b76cd.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr18_512x512_40k_voc12aug/fcn_hr18_512x512_40k_voc12aug_20200613_224401.log.json)     |
| FCN    | HRNetV2p-W48       | 512x512   |   40000 | -        | -              | 76.24 |         78.49 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr48_512x512_40k_voc12aug/fcn_hr48_512x512_40k_voc12aug_20200613_222111-1b0f18bc.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/hrnet/fcn_hr48_512x512_40k_voc12aug/fcn_hr48_512x512_40k_voc12aug_20200613_222111.log.json)     |
>>>>>>> mmseg/master
