### Another option: Docker Image

We provide a [Dockerfile](https://github.com/apulis/ApulisVision/blob/master/docker/Dockerfile) to build an image.


```shell
# build an image with PyTorch 1.5, CUDA 10.1
docker build -t apulisvision:1.5.0 docker/apulisvision/
```

Run it with

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/ApulisVision/data ApulisVision
```
