ARG PYTORCH="1.5"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compressgithub.com/open-mmlab/cocoap-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx sudo vim  \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
RUN pip install mmcv-full==latest+torch1.5.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html -i  https://pypi.tuna.tsinghua.edu.cn/simple
# Install ApulisVision
RUN conda clean --all
ENV FORCE_CUDA="1"
# 安装dev分支
RUN git clone -b dev --depth 1 https://github.com/apulis/ApulisVision.git /ApulisVision
WORKDIR /ApulisVision
RUN pip install -r requirements/production.txt -i  https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -r requirements/build.txt -i  https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -r requirements/optional.txt -i  https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"
RUN python setup_mmdet.py develop
RUN python setup_mmseg.py develop
RUN python setup_mmcls.py develop

