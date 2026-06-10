# syntax=docker/dockerfile:1.7
# ─────────────────────────────────────────────────────────────────────────────
#  LTX 2.3 / 10Eros — BAKED IMAGE
#  All models are downloaded at BUILD time. No network volume required.
#  Build with RunPod Build & Push (not GitHub Actions — runner disk is too small).
#
#  Required build secrets (set in RunPod builder or passed via --build-arg):
#    HF_TOKEN      — HuggingFace token (needed for gated TenStrip repos)
#    CIVITAI_TOKEN — Civitai API token (needed for LoRA downloads)
# ─────────────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_PREFER_BINARY=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONFAULTHANDLER=1
ENV SHELL=/bin/bash

# ─── System packages ──────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    git wget curl libgl1 libglib2.0-0 \
    libsm6 libxext6 libxrender1 ffmpeg \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# ─── ComfyUI via comfy-cli ────────────────────────────────────────────────────
RUN pip install --no-cache-dir comfy-cli
RUN /usr/bin/yes | comfy --workspace /comfyui install --nvidia

# ─── PyTorch (cu128) — reinstall after comfy-cli to ensure correct CUDA build ─
RUN pip install --no-cache-dir --force-reinstall \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# kornia pinned — ComfyUI-LTXVideo requires <=0.7.4
RUN pip install --no-cache-dir kornia==0.7.4

# ─── Custom nodes ─────────────────────────────────────────────────────────────
WORKDIR /comfyui

# 1. ComfyUI-LTXVideo (Lightricks official — all LTXV nodes)
RUN git clone --depth=1 \
    https://github.com/Lightricks/ComfyUI-LTXVideo.git \
    /comfyui/custom_nodes/ComfyUI-LTXVideo && \
    pip install --no-cache-dir \
    -r /comfyui/custom_nodes/ComfyUI-LTXVideo/requirements.txt && \
    pip install --no-cache-dir kornia==0.7.4

# 2. ComfyUI-Manager
RUN git clone --depth=1 \
    https://github.com/Comfy-Org/ComfyUI-Manager.git \
    /comfyui/custom_nodes/comfyui-manager && \
    pip install --no-cache-dir \
    -r /comfyui/custom_nodes/comfyui-manager/requirements.txt

# 3. ComfyMath (ComfyMathExpression — used for width/2 and height/2)
RUN git clone --depth=1 \
    https://github.com/evanspearman/ComfyMath.git \
    /comfyui/custom_nodes/ComfyMath
# ComfyMath is pure Python — no requirements.txt

# 4. KJNodes (ResizeImageMaskNode, ResizeImagesByLongerEdge)
RUN git clone --depth=1 \
    https://github.com/kijai/ComfyUI-KJNodes.git \
    /comfyui/custom_nodes/ComfyUI-KJNodes && \
    pip install --no-cache-dir \
    -r /comfyui/custom_nodes/ComfyUI-KJNodes/requirements.txt

# 5. ComfyUI-VideoHelperSuite (CreateVideo, SaveVideo)
RUN git clone --depth=1 \
    https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
    /comfyui/custom_nodes/ComfyUI-VideoHelperSuite && \
    pip install --no-cache-dir \
    -r /comfyui/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt

# ─── RunPod handler dependencies ──────────────────────────────────────────────
WORKDIR /
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Model directories ────────────────────────────────────────────────────────
RUN mkdir -p \
    /comfyui/models/checkpoints/10Eros \
    /comfyui/models/text_encoders/split_files/text_encoders \
    /comfyui/models/loras/ltxv/ltx2 \
    /comfyui/models/loras/ltxv/penile_praxis \
    /comfyui/models/loras/ltxv/anal_insertion \
    /comfyui/models/loras/ltxv/dr34ml4y \
    /comfyui/models/latent_upscale_models/LTX-Video \
    /comfyui/output \
    /comfyui/input

# ─── Bake models ──────────────────────────────────────────────────────────────
# Each model is a separate RUN layer so Docker layer cache can be reused
# on rebuilds if only one model changes. .tmp → mv pattern prevents partial
# files from being cached if a download is interrupted mid-build.

# Tokens are passed as build ARGs and never stored in the final image ENV
ARG HF_TOKEN
ARG CIVITAI_TOKEN

# --- 1/7  10Eros checkpoint (~30 GB) ---
# TenStrip/LTX2.3-10Eros is a gated HF repo — HF_TOKEN required
RUN wget -q --show-progress --progress=dot:giga \
    --header="Authorization: Bearer ${HF_TOKEN}" \
    -O /comfyui/models/checkpoints/10Eros/10Eros_v1-fp8mixed_learned.safetensors.tmp \
    "https://huggingface.co/TenStrip/LTX2.3-10Eros/resolve/main/10Eros_v1-fp8mixed_learned.safetensors" \
    && mv /comfyui/models/checkpoints/10Eros/10Eros_v1-fp8mixed_learned.safetensors.tmp \
          /comfyui/models/checkpoints/10Eros/10Eros_v1-fp8mixed_learned.safetensors \
    && echo "✓ 10Eros checkpoint: $(du -sh /comfyui/models/checkpoints/10Eros/10Eros_v1-fp8mixed_learned.safetensors | cut -f1)"

# --- 2/7  Gemma 3 12B text encoder fp8 (~13 GB) ---
# Comfy-Org/ltx-2 is public — HF_TOKEN still passed for rate-limit headroom
RUN wget -q --show-progress --progress=dot:giga \
    --header="Authorization: Bearer ${HF_TOKEN}" \
    -O /comfyui/models/text_encoders/split_files/text_encoders/Gemma_3_12B_it_fp8_scaled.safetensors.tmp \
    "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/Gemma_3_12B_it_fp8_scaled.safetensors" \
    && mv /comfyui/models/text_encoders/split_files/text_encoders/Gemma_3_12B_it_fp8_scaled.safetensors.tmp \
          /comfyui/models/text_encoders/split_files/text_encoders/Gemma_3_12B_it_fp8_scaled.safetensors \
    && echo "✓ Gemma text encoder: $(du -sh /comfyui/models/text_encoders/split_files/text_encoders/Gemma_3_12B_it_fp8_scaled.safetensors | cut -f1)"

# --- 3/7  Distilled cond-safe LoRA (~1 GB) ---
# TenStrip/LTX2.3_Distilled_Lora_1.1_Experiments — HF_TOKEN required
RUN wget -q --show-progress --progress=dot:giga \
    --header="Authorization: Bearer ${HF_TOKEN}" \
    -O /comfyui/models/loras/ltxv/ltx2/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors.tmp \
    "https://huggingface.co/TenStrip/LTX2.3_Distilled_Lora_1.1_Experiments/resolve/main/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors" \
    && mv /comfyui/models/loras/ltxv/ltx2/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors.tmp \
          /comfyui/models/loras/ltxv/ltx2/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors \
    && echo "✓ Distilled LoRA: $(du -sh /comfyui/models/loras/ltxv/ltx2/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors | cut -f1)"

# --- 4/7  Penile Praxis V4 LoRA (Civitai version 2772932) ---
# Civitai redirects to CDN — wget follows redirects automatically
RUN wget -q --show-progress --progress=dot:giga \
    --content-disposition \
    --header="Authorization: Bearer ${CIVITAI_TOKEN}" \
    -O /comfyui/models/loras/ltxv/penile_praxis/Penile_Praxis_V4.safetensors.tmp \
    "https://civitai.com/api/download/models/2772932" \
    && mv /comfyui/models/loras/ltxv/penile_praxis/Penile_Praxis_V4.safetensors.tmp \
          /comfyui/models/loras/ltxv/penile_praxis/Penile_Praxis_V4.safetensors \
    && echo "✓ Penile Praxis V4: $(du -sh /comfyui/models/loras/ltxv/penile_praxis/Penile_Praxis_V4.safetensors | cut -f1)"

# --- 5/7  Anal Insertion LoRA (Civitai version 2767135) ---
RUN wget -q --show-progress --progress=dot:giga \
    --content-disposition \
    --header="Authorization: Bearer ${CIVITAI_TOKEN}" \
    -O /comfyui/models/loras/ltxv/anal_insertion/nsfw_anal_insertion_ltx23_v1.0.safetensors.tmp \
    "https://civitai.com/api/download/models/2767135" \
    && mv /comfyui/models/loras/ltxv/anal_insertion/nsfw_anal_insertion_ltx23_v1.0.safetensors.tmp \
          /comfyui/models/loras/ltxv/anal_insertion/nsfw_anal_insertion_ltx23_v1.0.safetensors \
    && echo "✓ Anal Insertion LoRA: $(du -sh /comfyui/models/loras/ltxv/anal_insertion/nsfw_anal_insertion_ltx23_v1.0.safetensors | cut -f1)"

# --- 6/7  DR34ML4Y LoRA (Civitai version 2950842) ---
RUN wget -q --show-progress --progress=dot:giga \
    --content-disposition \
    --header="Authorization: Bearer ${CIVITAI_TOKEN}" \
    -O /comfyui/models/loras/ltxv/dr34ml4y/DR34ML4Y_LTXXX_V2.safetensors.tmp \
    "https://civitai.com/api/download/models/2950842" \
    && mv /comfyui/models/loras/ltxv/dr34ml4y/DR34ML4Y_LTXXX_V2.safetensors.tmp \
          /comfyui/models/loras/ltxv/dr34ml4y/DR34ML4Y_LTXXX_V2.safetensors \
    && echo "✓ DR34ML4Y LoRA: $(du -sh /comfyui/models/loras/ltxv/dr34ml4y/DR34ML4Y_LTXXX_V2.safetensors | cut -f1)"

# --- 7/7  Spatial upscaler (~950 MB) ---
# Lightricks/LTX-2.3 is public
RUN wget -q --show-progress --progress=dot:giga \
    --header="Authorization: Bearer ${HF_TOKEN}" \
    -O /comfyui/models/latent_upscale_models/LTX-Video/ltx-2.3-spatial-upscaler-x2-1.1.safetensors.tmp \
    "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors" \
    && mv /comfyui/models/latent_upscale_models/LTX-Video/ltx-2.3-spatial-upscaler-x2-1.1.safetensors.tmp \
          /comfyui/models/latent_upscale_models/LTX-Video/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
    && echo "✓ Spatial upscaler: $(du -sh /comfyui/models/latent_upscale_models/LTX-Video/ltx-2.3-spatial-upscaler-x2-1.1.safetensors | cut -f1)"

# ─── Verify baked model inventory ─────────────────────────────────────────────
RUN echo "=== BAKED MODEL INVENTORY ===" && \
    find /comfyui/models -name "*.safetensors" | sort | \
    while IFS= read -r f; do \
        size=$(du -sh "$f" | cut -f1); \
        echo "  $size  $f"; \
    done && \
    echo "=== END INVENTORY ==="

# ─── extra_model_paths.yaml ───────────────────────────────────────────────────
# Baked image: models live at /comfyui/models — no volume path needed.
COPY extra_model_paths.yaml /comfyui/extra_model_paths.yaml

# ─── Application files ────────────────────────────────────────────────────────
COPY src/start.sh /src/start.sh
COPY handler.py ltx_payload_builder.py workflow_support.py \
     video_ltx23_10eros_i2v_API.json ./
RUN chmod +x /src/start.sh

CMD ["/src/start.sh"]
