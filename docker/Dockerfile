ARG PYTORCH="1.5"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y git vim ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
RUN pip install mmcv-full==latest+torch1.5.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html

# Install ApulisVision
RUN conda clean --all
RUN git clone https://github.com/apulis/ApulisVision.git /ApulisVision
WORKDIR /ApulisVision
ENV FORCE_CUDA="1"
RUN pip install cython
RUN pip install scipy
RUN pip install scikit-image
RUN pip install tensorboard
RUN pip install -r requirements/build.txt
RUN pip install -r requirements/optional.txt
RUN pip install jupyterlab
RUN pip install jupyter
RUN pip install --no-cache-dir -e .
