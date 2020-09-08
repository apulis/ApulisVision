yapf -r -i mmdet/ mmcls/ mmseg/ configs/ tests/mmdet/ tools/
isort -rc mmdet/ mmcls/ mmseg/ configs/ tests/mmdet/ tools/
flake8 .
