ARG PYTORCH="1.5"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx  \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
RUN pip install mmcv-full==latest+torch1.5.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html -i  https://pypi.tuna.tsinghua.edu.cn/simple
# Install ApulisVision
RUN conda clean --all
ENV FORCE_CUDA="1"
RUN pip install cython -i  https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install scipy -i  https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install scikit-image -i  https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install tensorboard -i  https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -r requirements/build.txt -i  https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -r requirements/optional.txt -i  https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install jupyterlab -i  https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install mmcv-full==1.1.1 -i  https://pypi.tuna.tsinghua.edu.cn/simple

# 安装dev分支
RUN git clone -b dev https://github.com/apulis/ApulisVision.git /ApulisVision
WORKDIR /ApulisVision
RUN python setup_mmdet.py develop
RUN python setup_mmseg.py develop
RUN python setup_mmcls.py develop

