#!/bin/bash
set -euo pipefail

# ─── Logging helpers ──────────────────────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%dT%H:%M:%S')] $*"; }

# MODEL_STORE is set by start.sh and points to the volume (or local fallback)
COMFY_MODELS="${MODEL_STORE:-/comfyui/models}"
HF_TOKEN="${HF_TOKEN:-}"
CIVITAI_TOKEN="${CIVITAI_TOKEN:-}"

log "=== download_models.sh starting ==="
log "COMFY_MODELS = $COMFY_MODELS"
log "HF_TOKEN     = ${HF_TOKEN:+set (${#HF_TOKEN} chars)}${HF_TOKEN:-NOT SET}"
log "CIVITAI_TOKEN= ${CIVITAI_TOKEN:+set (${#CIVITAI_TOKEN} chars)}${CIVITAI_TOKEN:-NOT SET}"

# ─── Download helpers ─────────────────────────────────────────────────────────

hf_download() {
    local url="$1"
    local dest="$2"
    local name
    name="$(basename "$dest")"

    mkdir -p "$(dirname "$dest")"

    if [ -f "$dest" ] && [ -s "$dest" ]; then
        local size
        size=$(du -sh "$dest" | cut -f1)
        log "[SKIP] $name already present ($size)"
        return 0
    fi

    log "[DL] HuggingFace: $name"
    log "     → $url"
    log "     → $dest"

    local start_time=$SECONDS
    wget --progress=dot:giga \
        --header="Authorization: Bearer $HF_TOKEN" \
        -O "${dest}.tmp" "$url" 2>&1 | \
        while IFS= read -r line; do log "     $line"; done || true

    if [ ! -f "${dest}.tmp" ] || [ ! -s "${dest}.tmp" ]; then
        log "[ERROR] Download produced empty file: $dest"
        rm -f "${dest}.tmp"
        exit 1
    fi

    mv "${dest}.tmp" "$dest"
    local elapsed=$(( SECONDS - start_time ))
    local size
    size=$(du -sh "$dest" | cut -f1)
    log "[OK]  $name — ${size} in ${elapsed}s"
}

civitai_download() {
    local version_id="$1"
    local dest="$2"
    local name
    name="$(basename "$dest")"

    mkdir -p "$(dirname "$dest")"

    if [ -f "$dest" ] && [ -s "$dest" ]; then
        local size
        size=$(du -sh "$dest" | cut -f1)
        log "[SKIP] $name already present ($size)"
        return 0
    fi

    if [ -z "$CIVITAI_TOKEN" ]; then
        log "[ERROR] CIVITAI_TOKEN not set — cannot download $name (version $version_id)"
        exit 1
    fi

    log "[DL] Civitai version $version_id: $name"
    log "     → $dest"

    local start_time=$SECONDS
    wget --progress=dot:giga \
        --header="Authorization: Bearer $CIVITAI_TOKEN" \
        -O "${dest}.tmp" \
        "https://civitai.com/api/download/models/$version_id" 2>&1 | \
        while IFS= read -r line; do log "     $line"; done || true

    if [ ! -f "${dest}.tmp" ] || [ ! -s "${dest}.tmp" ]; then
        log "[ERROR] Download produced empty file: $dest"
        rm -f "${dest}.tmp"
        exit 1
    fi

    mv "${dest}.tmp" "$dest"
    local elapsed=$(( SECONDS - start_time ))
    local size
    size=$(du -sh "$dest" | cut -f1)
    log "[OK]  $name — ${size} in ${elapsed}s"
}

# ─── Model downloads ──────────────────────────────────────────────────────────

log "--- 1/7  10Eros checkpoint (~30 GB) ---"
hf_download \
    "https://huggingface.co/TenStrip/LTX2.3-10Eros/resolve/main/10Eros_v1-fp8mixed_learned.safetensors" \
    "$COMFY_MODELS/checkpoints/10Eros/10Eros_v1-fp8mixed_learned.safetensors"

log "--- 2/7  Gemma 3 12B text encoder fp8 (~13 GB) ---"
hf_download \
    "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/Gemma_3_12B_it_fp8_scaled.safetensors" \
    "$COMFY_MODELS/text_encoders/split_files/text_encoders/Gemma_3_12B_it_fp8_scaled.safetensors"

log "--- 3/7  Distilled cond-safe LoRA (~1 GB) ---"
hf_download \
    "https://huggingface.co/TenStrip/LTX2.3_Distilled_Lora_1.1_Experiments/resolve/main/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors" \
    "$COMFY_MODELS/loras/ltxv/ltx2/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors"

log "--- 4/7  Penile Praxis V4 LoRA (Civitai 2772932) ---"
civitai_download "2772932" \
    "$COMFY_MODELS/loras/ltxv/penile_praxis/Penile_Praxis_V4.safetensors"

log "--- 5/7  Anal Insertion LoRA (Civitai 2767135) ---"
civitai_download "2767135" \
    "$COMFY_MODELS/loras/ltxv/anal_insertion/nsfw_anal_insertion_ltx23_v1.0.safetensors"

log "--- 6/7  DR34ML4Y LoRA (Civitai 2950842) ---"
civitai_download "2950842" \
    "$COMFY_MODELS/loras/ltxv/dr34ml4y/DR34ML4Y_LTXXX_V2.safetensors"

log "--- 7/7  Spatial upscaler (~950 MB) ---"
hf_download \
    "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors" \
    "$COMFY_MODELS/latent_upscale_models/LTX-Video/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

# ─── Final inventory ──────────────────────────────────────────────────────────
log "=== Download complete. Model inventory ==="
find "$COMFY_MODELS" -type f -name "*.safetensors" | sort | \
    while IFS= read -r f; do
        size=$(du -sh "$f" | cut -f1)
        log "  $size  $f"
    done

log "=== download_models.sh done ==="
