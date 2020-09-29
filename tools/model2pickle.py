# Import the required modules
import sys
import os.path as osp
# 鏡像新增
import cloudpickle as pickle

from mmdet.apis import inference_detector, init_detector
from mmcls.apis import inference_classfication, init_classfication
from mmseg.apis import inference_segmentor, init_segmentor

target_type = sys.argv


def pickle_dump(obj, file):
    pickled_lambda = pickle.dumps(obj)
    with open(file, "wb") as f:
        f.write(pickled_lambda)


def pickle_load(file):
    with open(file, "rb") as f:
        model_s = f.read()
    return pickle.loads(model_s)


class InferenceModel:
    def __init__(self, model, predict):
        self.model = model
        self.predict = predict


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
