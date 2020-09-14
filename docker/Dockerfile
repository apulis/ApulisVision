ARG PYTORCH="1.5"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN apt-get update && apt-get install -y git vim ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
RUN pip install mmcv-full==latest+torch1.5.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html -i  https://pypi.tuna.tsinghua.edu.cn/simple
# Install ApulisVision
RUN conda clean --all
#RUN git clone https://github.com/apulis/ApulisVision.git /ApulisVision
RUN COPY ../ApulisVision /ApulisVision
WORKDIR /ApulisVision
ENV FORCE_CUDA="1"
RUN pip install cython
RUN pip install scipy -i  https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install scikit-image -i  https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install tensorboard -i  https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -r requirements/build.txt -i  https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -r requirements/optional.txt -i  https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install jupyterlab -i  https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install jupyter -i  https://pypi.tuna.tsinghua.edu.cn/simple
