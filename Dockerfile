# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_PREFER_BINARY=1
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# Install comfy-cli and ComfyUI
RUN pip install --no-cache-dir comfy-cli
RUN /usr/bin/yes | comfy --workspace /comfyui install --nvidia

# Upgrade PyTorch to cu124
RUN pip install --no-cache-dir --force-reinstall \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# Fix kornia version (ComfyUI-LTXVideo requires 0.7.4)
RUN pip install --no-cache-dir kornia==0.7.4

WORKDIR /comfyui

# ComfyUI-LTXVideo (Lightricks official nodes)
RUN git clone --depth=1 \
    https://github.com/Lightricks/ComfyUI-LTXVideo.git \
    /comfyui/custom_nodes/ComfyUI-LTXVideo && \
    pip install --no-cache-dir \
    -r /comfyui/custom_nodes/ComfyUI-LTXVideo/requirements.txt && \
    pip install --no-cache-dir kornia==0.7.4

# ComfyUI-Manager
RUN git clone --depth=1 \
    https://github.com/Comfy-Org/ComfyUI-Manager.git \
    /comfyui/custom_nodes/comfyui-manager && \
    pip install --no-cache-dir \
    -r /comfyui/custom_nodes/comfyui-manager/requirements.txt

# ComfyMath (ComfyMathExpression nodes)
RUN git clone --depth=1 \
    https://github.com/evanspearman/ComfyMath.git \
    /comfyui/custom_nodes/ComfyMath

# Handler dependencies
WORKDIR /
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- VARIABLES & MODEL DOWNLOAD PHASE ---
# Declare arguments right before usage to prevent build-cache isolation blanking
ARG HF_TOKEN
ARG CIVITAI_TOKEN

# Map them to active runtime system environments for the shell downloader execution block
ENV HF_TOKEN=${HF_TOKEN}
ENV CIVITAI_TOKEN=${CIVITAI_TOKEN}

# Download all models at build time
COPY src/download_models.sh /src/download_models.sh
RUN chmod +x /src/download_models.sh && \
    bash /src/download_models.sh

# App files
COPY src/start.sh /src/start.sh
COPY handler.py ltx_payload_builder.py workflow_support.py \
     video_ltx23_10eros_i2v_API.json ./

RUN chmod +x /src/start.sh

CMD ["/src/start.sh"]