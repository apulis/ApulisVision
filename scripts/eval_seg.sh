#!/bin/sh

python tools/eval_seg.py --config configs_custom/mmseg/fcn_r50-d8_512x1024_40k_cityscapes.py \
        --pipeline_config panel/pipeline_seg_panel.json \
        --eval bbox
