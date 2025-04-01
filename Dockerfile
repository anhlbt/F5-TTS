# FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel


USER root

ARG DEBIAN_FRONTEND=noninteractive


RUN set -x \
    && apt-get update \
    && apt-get -y install wget curl man git less openssl libssl-dev unzip unar build-essential aria2 tmux vim \
    && apt-get install -y openssh-server sox libsox-fmt-all libsox-fmt-mp3 libsndfile1-dev ffmpeg \
    && apt-get install -y librdmacm1 libibumad3 librdmacm-dev libibverbs1 libibverbs-dev ibverbs-utils ibverbs-providers \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /workspace

# RUN git clone https://github.com/SWivid/F5-TTS.git \
#     && cd F5-TTS \
#     && git submodule update --init --recursive \
#     && pip install -e . --no-cache-dir
COPY . ./F5-TTS/
RUN  cd F5-TTS && pip install -e . --no-cache-dir

RUN cd F5-TTS && pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /workspace/F5-TTS/outputs
ENV SHELL=/bin/bash
# RUN pip install tensorboard cached_path
WORKDIR /workspace/F5-TTS
