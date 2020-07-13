# Image Classification

## Introduction

Welcome to the Image-Classification part, we have implemented the image-classification train pipeline based on [mmdetection](https://github.com/open-mmlab/mmdetection).


## Main Support

Following models are implemented using PyTorch.

* ResNet ([1512.03385](https://arxiv.org/abs/1512.03385))
* ResNeXt ([1611.05431](https://arxiv.org/abs/1611.05431))
* SENet ([1709.01507](https://arxiv.org/abs/1709.01507))
* Res2Net ([1904.01169](https://arxiv.org/abs/1904.01169))
* RegNet ([/2003.13678](https://arxiv.org/pdf/2003.13678.pdf))

To do list:
* VGG
* GoogleNet
* MobileNet
* EfficientNet
* Xception

## Get Started
Please see [GETTING_STARTED.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/getting_started.md) for the basic usage of MMDetection.

### Prepare datasets

For image-classificatin, you need to prepare your datasets in the following formate. 
If your folder structure is different, you may need to change the corresponding paths in config files. 

```plain
hymenoptera_data/
├── train
│   ├── ants
│   └── bees
├── val
│   ├── ants
│   └── bees
```

## Train a model
### Train with a single GPU

```shell
python tools/train.py ${CONFIG_FILE}
```

If you want to specify the working directory in the command, you can add an argument `--work_dir ${YOUR_WORK_DIR}`.

### Train with multiple GPUs

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

### Test 

You can use the following commands to test a dataset.

```shell
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show]

# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]
```