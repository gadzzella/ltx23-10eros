# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_PREFER_BINARY=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    git wget curl libgl1 libglib2.0-0 \
    libsm6 libxext6 libxrender1 ffmpeg \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir comfy-cli
RUN /usr/bin/yes | comfy --workspace /comfyui install --nvidia

RUN pip install --no-cache-dir --force-reinstall \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir kornia==0.7.4

WORKDIR /comfyui

RUN git clone --depth=1 \
    https://github.com/Lightricks/ComfyUI-LTXVideo.git \
    /comfyui/custom_nodes/ComfyUI-LTXVideo && \
    pip install --no-cache-dir \
    -r /comfyui/custom_nodes/ComfyUI-LTXVideo/requirements.txt && \
    pip install --no-cache-dir kornia==0.7.4

RUN git clone --depth=1 \
    https://github.com/Comfy-Org/ComfyUI-Manager.git \
    /comfyui/custom_nodes/comfyui-manager && \
    pip install --no-cache-dir \
    -r /comfyui/custom_nodes/comfyui-manager/requirements.txt

RUN git clone --depth=1 \
    https://github.com/evanspearman/ComfyMath.git \
    /comfyui/custom_nodes/ComfyMath

WORKDIR /
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App files only — models download at runtime via start.sh
COPY src/start.sh /src/start.sh
COPY src/download_models.sh /src/download_models.sh
COPY handler.py ltx_payload_builder.py workflow_support.py \
     video_ltx23_10eros_i2v_API.json ./

RUN chmod +x /src/start.sh /src/download_models.sh

CMD ["/src/start.sh"]
