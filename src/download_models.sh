#!/bin/bash
set -euo pipefail

COMFY_MODELS="/comfyui/models"
HF_TOKEN="${HF_TOKEN:-}"
CIVITAI_TOKEN="${CIVITAI_TOKEN:-}"

hf_download() {
    local url="$1"
    local dest="$2"
    mkdir -p "$(dirname "$dest")"
    if [ -f "$dest" ] && [ -s "$dest" ]; then
        echo "[SKIP] $dest"
        return 0
    fi
    echo "[DL] $url"
    wget --quiet --show-progress \
        --header="Authorization: Bearer $HF_TOKEN" \
        -O "$dest" "$url"
    if [ ! -s "$dest" ]; then
        echo "[ERROR] Download failed or empty: $dest"
        exit 1
    fi
}

civitai_download() {
    local version_id="$1"
    local dest="$2"
    mkdir -p "$(dirname "$dest")"
    if [ -f "$dest" ] && [ -s "$dest" ]; then
        echo "[SKIP] $dest"
        return 0
    fi
    if [ -z "$CIVITAI_TOKEN" ]; then
        echo "[ERROR] CIVITAI_TOKEN not set"
        exit 1
    fi
    echo "[DL] Civitai $version_id -> $dest"
    wget --quiet --show-progress \
        --header="Authorization: Bearer $CIVITAI_TOKEN" \
        -O "$dest" \
        "https://civitai.com/api/download/models/$version_id"
    if [ ! -s "$dest" ]; then
        echo "[ERROR] Download failed or empty: $dest"
        exit 1
    fi
}

echo "=== Downloading models ==="

# 1. 10Eros checkpoint (~30GB) — bundles video VAE + audio VAE
hf_download \
    "https://huggingface.co/TenStrip/LTX2.3-10Eros/resolve/main/10Eros_v1-fp8mixed_learned.safetensors" \
    "$COMFY_MODELS/checkpoints/10Eros/10Eros_v1-fp8mixed_learned.safetensors"

# 2. Gemma 3 12B text encoder fp8 (~13GB)
hf_download \
    "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/Gemma_3_12B_it_fp8_scaled.safetensors" \
    "$COMFY_MODELS/text_encoders/split_files/text_encoders/Gemma_3_12B_it_fp8_scaled.safetensors"

# 3. Distilled cond-safe LoRA (~1GB)
hf_download \
    "https://huggingface.co/TenStrip/LTX2.3_Distilled_Lora_1.1_Experiments/resolve/main/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors" \
    "$COMFY_MODELS/loras/ltxv/ltx2/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors"

# 4. Penile Praxis LoRA (Civitai 2772932)
civitai_download "2772932" \
    "$COMFY_MODELS/loras/ltxv/penile_praxis/Penile_Praxis_V4.safetensors"

# 5. Anal Insertion LoRA (Civitai 2767135)
civitai_download "2767135" \
    "$COMFY_MODELS/loras/ltxv/anal_insertion/nsfw_anal_insertion_ltx23_v1.0.safetensors"

# 6. DR34ML4Y LoRA (Civitai 2950842)
civitai_download "2950842" \
    "$COMFY_MODELS/loras/ltxv/dr34ml4y/DR34ML4Y_LTXXX_V2.safetensors"

# 7. Spatial upscaler (~950MB)
hf_download \
    "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors" \
    "$COMFY_MODELS/latent_upscale_models/LTX-Video/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

echo "=== All models downloaded successfully ==="
