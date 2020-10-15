import kfserving
from typing import List, Dict
import torch
from PIL import Image
import base64
import io
import cloudpickle as pickle
import numpy as np
import os


def pickle_load(file):
    with open(file, 'rb') as f:
        model_s = f.read()
    return pickle.loads(model_s)

class KFServingSampleModel(kfserving.KFModel):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False

    def load(self):
        self.infer_model = pickle_load(os.path.join("/mnt/models","export_model.pkl"))
        self.ready = True

    def model_infer(self, inputImg):
        inputImg = np.asarray(inputImg)
        result = self.infer_model.predict(self.infer_model.model, inputImg)
        return result

    def predict(self, request: Dict) -> Dict:
        inputs = request["instances"]
        output = self.model_infer(inputs)
        return {"predictions": output}


if __name__ == "__main__":
    model = KFServingSampleModel("kfserving-custom-model")
    model.load()
    kfserving.KFServer(workers=1).start([model])