FROM nvcr.io/nvidia/tensorrt:21.09-py3

ENV DEBIAN_FRONTEND=noninteractive
ARG WORKDIR=/workspace/safety-detection

RUN apt-get clean && apt-get update && apt-get install -y \
        automake autoconf libpng-dev nano python3-pip \
        curl zip unzip libtool swig zlib1g-dev pkg-config \
        python3-mock libpython3-dev libpython3-all-dev \
        g++ gcc cmake make pciutils cpio gosu wget \
        libgtk-3-dev libxtst-dev sudo apt-transport-https \
        build-essential gnupg git xz-utils vim ffmpeg python3-tk \
        libva-drm2 libva-x11-2 vainfo libva-wayland2 libva-glx2 \
        libva-dev libdrm-dev xorg xorg-dev protobuf-compiler \
        openbox libx11-dev libgl1-mesa-glx libgl1-mesa-dev \
        libtbb2 libtbb-dev libopenblas-dev libopenmpi-dev \
    && sed -i 's/# set linenumbers/set linenumbers/g' /etc/nanorc \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

COPY ./safety-detection/requirements.txt /install/requirements.txt

RUN python -m pip install --upgrade pip && python -m pip install -r /install/requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

RUN MMCV_WITH_OPS=1 mim install mmcv-full==1.6.0

WORKDIR ${WORKDIR}

# docker build . -t alex:latest
# docker run --gpus all -it -v $(pwd)/safety-detection:/workspace/safety-detection --rm alex:latest
# have to run this command "cd mmsegmentation && pip install -v -e ."