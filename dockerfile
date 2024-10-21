FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

RUN pip install requirements.txt