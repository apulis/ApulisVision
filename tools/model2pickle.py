# Import the required modules
import os.path as osp
import sys

# 鏡像新增
import cloudpickle as pickle
<<<<<<< HEAD
=======

from mmdet.apis import inference_detector, init_detector
>>>>>>> a3c6197244216d43abb3c4e286c2c1e2989dcb2a
from mmcls.apis import inference_classfication, init_classfication
from mmseg.apis import inference_segmentor, init_segmentor

target_type = sys.argv

from mmdet.apis import inference_detector, init_detector, show_result_pyplot


def pickle_dump(obj, file):
    pickled_lambda = pickle.dumps(obj)
    with open(file, 'wb') as f:
        f.write(pickled_lambda)


def pickle_load(file):
    with open(file, 'rb') as f:
        model_s = f.read()
    return pickle.loads(model_s)


class InferenceModel:

    def __init__(self, model, predict):
        self.model = model
        self.predict = predict


<<<<<<< HEAD
def det_infer():
    config_file = '/home/kaiyuan.xu/ApulisVision/configs_custom/mmdet/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = '/home/kaiyuan.xu/ApulisVision/work_dir/epoch_1.pth'
    img = '../demo/demo.jpg'
    model = init_detector(
        config=config_file, checkpoint=checkpoint_file, device='cuda:0')
    infer_model = InferenceModel(model, inference_detector)
    pickle_dump(infer_model,
                osp.join(osp.dirname(osp.abspath(__file__)), 'model.pkl'))
    infer_model = pickle_load(
        osp.join(osp.dirname(osp.abspath(__file__)), 'model.pkl'))
    result = infer_model.predict(infer_model.model, img)
    print(result[0][0])


def seg_infer():
    config_file = '/home/kaiyuan.xu/ApulisVision/configs_custom/mmdet/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = '/home/kaiyuan.xu/ApulisVision/work_dir/epoch_1.pth'
    img = '../demo/demo.jpg'
    model = init_segmentor(
        config=config_file, checkpoint=checkpoint_file, device='cuda:0')
    infer_model = InferenceModel(model, inference_segmentor)
    pickle_dump(infer_model,
                osp.join(osp.dirname(osp.abspath(__file__)), 'model.pkl'))
    infer_model = pickle_load(
        osp.join(osp.dirname(osp.abspath(__file__)), 'model.pkl'))
    result = infer_model.predict(infer_model.model, img)
    print(result[0][0])


def cls_infer():
    config_file = '/home/kaiyuan.xu/ApulisVision/configs_custom/mmdet/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = '/home/kaiyuan.xu/ApulisVision/work_dir/epoch_1.pth'
    img = '../demo/demo.jpg'
    model = init_classfication(
        config=config_file, checkpoint=checkpoint_file, device='cuda:0')
    infer_model = InferenceModel(model, inference_classfication())
    pickle_dump(infer_model,
                osp.join(osp.dirname(osp.abspath(__file__)), 'model.pkl'))
    infer_model = pickle_load(
        osp.join(osp.dirname(osp.abspath(__file__)), 'model.pkl'))
    result = infer_model.predict(infer_model.model, img)
    print(result[0][0])


det_infer()
seg_infer()
cls_infer()
=======
def save_image(data):
    imgname = "temp.jpg"
    with open(imgname, 'wb')as f:
        f.write(data)
    return imgname


def dump_infer_model(checkpoint_file, config_file, output_file, target='det', device='cuda:0'):
    print("------------------------------")
    print("START EXPORT MODEL")
    print("------------------------------")
    if target == 'det':
        model = init_detector(config=config_file, checkpoint=checkpoint_file, device=device)
        infer_model = InferenceModel(model, inference_detector)
    elif target == 'cls':
        model = init_classfication(config=config_file, checkpoint=checkpoint_file, device=device)
        infer_model = InferenceModel(model, inference_classfication)
    elif target == 'seg':
        model = init_segmentor(config=config_file, checkpoint=checkpoint_file, device=device)
        infer_model = InferenceModel(model, inference_segmentor)
    pickle_dump(infer_model, output_file)
    model_infer(output_file)
    print("------------------------------")
    print("SUCCESS EXPORT MODEL")
    print("------------------------------")


def model_infer(output_file):
    img = '../demo/demo.jpg'
    infer_model = pickle_load(output_file)
    result = infer_model.predict(infer_model.model, img)
    print(result)
>>>>>>> a3c6197244216d43abb3c4e286c2c1e2989dcb2a
