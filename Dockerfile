FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

COPY ./safety-detection/requirements.txt /install/requirements.txt

WORKDIR /workspace/safety-detection

RUN apt-get update && apt-get install -y \
		ffmpeg libsm6 libxext6 \
		&& apt-get install gcc -y 

RUN python -m pip install -r /install/requirements.txt

RUN MMCV_WITH_OPS=1 mim install mmcv-full

# docker run --gpus all -it -v $(pwd)/safety-detection:/workspace/safety-detection --rm alex:latest
# have to run this command "cd mmsegmentation && pip install -v -e ."