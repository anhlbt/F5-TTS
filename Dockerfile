# WORKDIR /workspace

# RUN git clone https://github.com/SWivid/F5-TTS.git \
#     && cd F5-TTS \
#     && git submodule update --init --recursive \
#     && pip install -e . --no-cache-dir
# Use PyTorch with CUDA support as the base image
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

USER root

# Set non-interactive frontend for apt
ARG DEBIAN_FRONTEND=noninteractive

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies, including fixes for Calibre
RUN set -x \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    wget curl man git less openssl libssl-dev unzip unar build-essential aria2 tmux vim \
    openssh-server sox libsox-fmt-all libsox-fmt-mp3 libsndfile1-dev ffmpeg \
    librdmacm1 libibumad3 librdmacm-dev libibverbs1 libibverbs-dev ibverbs-utils ibverbs-providers \
    libegl1 libgl1 libopengl0 libxcb-cursor0 libxcb-shape0 libxcb-randr0 libxcb-render0 \
    libxcb-render-util0 libxcb-image0 libxcb-keysyms1 libxcb-glx0 libxkbcommon0 \
    libxkbcommon-x11-0 libx11-xcb1 libssl3 libxml2 libxslt1.1 libsqlite3-0 zlib1g \
    libopenjp2-7 libjpeg-turbo8 libpng16-16 libtiff-dev libwebp7 poppler-utils \
    libxml2-dev libxslt1-dev \
    bash-completion xdg-utils libxcomposite1 \
    qt6-base-dev qt6-webengine-dev libqt6webenginecore6 libqt6webenginewidgets6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Calibre for ebook conversion
RUN wget -nv -O- https://download.calibre-ebook.com/linux-installer.sh | sh /dev/stdin

# Ensure Calibre's ebook-convert is in PATH
ENV PATH="/opt/calibre:${PATH}"

# Set working directory
WORKDIR /workspace/F5-TTS

# Create necessary directories
RUN mkdir -p /workspace/F5-TTS/outputs \
    /workspace/F5-TTS/tools/ebook/Working_files/Book \
    /workspace/F5-TTS/tools/ebook/Working_files/temp_ebook \
    /workspace/F5-TTS/tools/ebook/Working_files/temp

# Copy project files (assuming F5-TTS source is in the current directory)
# COPY . .

#--no-cache-dir
COPY requirements_audio-separator.txt ./
COPY requirements.txt ./
RUN pip install -r requirements_audio-separator.txt --no-deps
RUN pip install -r requirements.txt

# Upgrade pip and install Python dependencies
COPY . .
RUN pip install --upgrade pip \
    && pip install -e .

RUN pip install markitdown[all]
RUN pip install rnnoise
# Install NLTK data
RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt \
    && echo "Listing NLTK punkt data directory contents:" \
    && ls -lR /usr/local/share/nltk_data/tokenizers/punkt || echo "Punkt directory not found or ls failed"

# Set NLTK data path
ENV NLTK_DATA=/usr/local/share/nltk_data:/usr/share/nltk_data:/root/nltk_data


# Set shell
ENV SHELL=/bin/bash

# Expose port
EXPOSE 80