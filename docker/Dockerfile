ARG PYTORCH="2.1.0"
ARG CUDA="12.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

WORKDIR /root/workspace

COPY docker/requirements.txt /root/workspace/docker/

RUN pip install --no-cache-dir -r docker/requirements.txt

COPY . /root/workspace/

ENTRYPOINT ["python", "train.py", "-c", "configs/config.yaml"]



