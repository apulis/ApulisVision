#!/bin/sh

python tools/eval_det.py --config configs_custom/mmdet/faster_rcnn_r50_fpn_1x_coco.py \
        --pipeline_config panel/pipeline_det_panel.json \
        --eval bbox
