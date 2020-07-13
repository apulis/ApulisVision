
# Semantic-Segmentation

## Introduction

Welcome to the Semantic-Segmentation, we have implemented the Semantic-Segmentation train pipeline based on [mmdetection](https://github.com/open-mmlab/mmdetection).

## Supported Models

Following models are implemented using PyTorch. 

- [DeepLabv3](https://arxiv.org/abs/1706.05587.pdf)
- [DeepLabv3+](https://arxiv.org/pdf/1802.02611.pdf)
- [DenseASPP](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.pdf)
- [Unet](http://www.arxiv.org/pdf/1505.04597.pdf)
- [PSPNet](https://arxiv.org/pdf/1612.01105.pdf)
- [PSANet](https://hszhao.github.io/papers/eccv18_psanet.pdf)
- [EMANet](https://arxiv.org/pdf/1907.13426.pdf)
- [DANet](https://arxiv.org/pdf/1809.02983.pdf)
- [CCNet](https://arxiv.org/pdf/1811.11721.pdf)
- [OCR](https://arxiv.org/pdf/1909.11065.pdf)


## Get Started
Please see [GETTING_STARTED.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/getting_started.md) for the basic usage of ApulisVison.


## Experiments on DeepGlobe Landcover Dataset

### On Local Subset
|Methods|Backbone|Norm Layer|Epochs|Loss|Optimizer|Weight Decay|Multi-test|Mean IoU|PixAcc|Use mmdetection|Comments|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|DeepLabv3|resnet50(dlt)|SyncBN|130|focal + lovasz|sgd|0.001|no|0.5994|0.8636|||
|DeepLabv3+|xception65|SyncBN|130|focal + lovasz|sgd|0.001|no|0.4452|0.7867|||
|DeepAggregationNet|resnet50|SyncBN|100|focal + lovasz|sgd|0.001|no|0.4213||√||
|FPN|resnet50|SyncBN|60|focal + lovasz|sgd|0.001|no|0.4384||√||
|Unet with SE module|resnet50|SyncBN|60|focal + lovasz|sgd|0.001|no|0.4237||√||
|DeepLabv3|resnet50(dlt)|SyncBN|85|focal + lovasz|sgd|0.0005|no|0.6000|0.8594|√||
|DeepLabv3|resnet50(dlt)|SyncBN|85|focal + lovasz|sgd|0.0005|5 augs|0.6074|0.8666|√||
|DeepAggregationNet|resnet50(dlt)|SyncBN|85|focal + lovasz|sgd|0.0005|no|0.6006|0.8640|√|smaller gap between train & val|
|DeepAggregationNet|resnet50(dlt)|SyncBN|85|focal + lovasz|sgd|0.0005|5 augs|0.6069|0.8661|√||
|DeepAggregationNet|resnet50(dlt)|SyncBN|100|focal + lovasz|sgd|0.0005|no|0.6186|0.8648|√|reduce shallow features to 48 channels|
|DenseASPP|resnet50(dlt)|SyncBN|100|focal + lovasz|sgd|0.0005|no|0.7060|0.8766|√||
|DenseASPP|resnet50(dlt)|SyncBN|100|focal + lovasz|sgd|0.0005|5 augs|0.7024|0.8792|√||
|DenseASPP|resnet50(jpu)|SyncBN|100|focal + lovasz|sgd|0.0005|no|0.6949|0.8736|√|lower computational complexity than dilated convolution|
|DenseASPP|resnet50(jpu)|SyncBN|100|focal + lovasz|sgd|0.0005|5 augs|0.6962|0.8794|√||
|DenseASPP+CARAFE|resnet50(dlt)|SyncBN|100|focal + lovasz|sgd|0.0005|no|0.6050|0.8221|√||
|OCR|hrnet32|SyncBN|100|focal + lovasz|sgd|0.0005|no|0.6913|0.8709|√||
|OCR|hrnet32|SyncBN|100|focal + lovasz|sgd|0.0005|5 augs|0.6890|0.8747|√||

### Comparison by category
|Methods|Urban land|Agriculture land|Range land|Forest land|Water|Barren land|Unknown|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|DeepLabv3|0.7319|0.8569|0.4098|0.7258|0.7878|0.6461|0.0421|
|DeepAggregationNet(128 channels)|0.7486|0.8464|0.3845|0.7637|0.7589|0.6403|0.0618|
|DeepAggregationNet(48 channels)|0.7595|0.8560|0.3970|0.7598|0.7963|0.6148|0.1469|
|DenseASPP(dlt)|0.7785|0.8757|0.4238|0.7496|0.8196|0.6752|0.6198|
|DenseASPP(jpu)|0.7305|0.8708|0.4254|0.7637|0.8019|0.6688|0.6034|
|OCR|0.7616|0.8572|0.4427|0.7650|0.7739|0.6901|0.5482|

### On Leaderboard
|Methods|Backbone|Norm Layer|Epochs|Loss|Optimizer|Multi-test|Mean IoU|Use mmdetection|Comments|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|DeepLabv3|resnet50|BN|100|focal + lovasz|sgd|no|0.4905|||
|DeepLabv3+|resnet50|SyncBN|100|focal + lovasz|sgd|no|0.4615|||