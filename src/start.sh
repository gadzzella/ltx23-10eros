#!/bin/bash
set -euo pipefail

# ─── Logging helpers ──────────────────────────────────────────────────────────
ts()  { date '+%Y-%m-%dT%H:%M:%S'; }
log() { echo "[$(ts())] $*"; }
sep() { echo "[$(ts())] ═══════════════════════════════════════════════════"; }

sep
log "=== LTX 2.3 / 10Eros Worker Starting ==="
log "Hostname      : $(hostname)"
log "CUDA devices  : ${CUDA_VISIBLE_DEVICES:-<not set>}"
log "NVIDIA SMI    :"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null \
    | while IFS= read -r line; do log "  GPU: $line"; done || log "  (nvidia-smi unavailable)"
sep

# ─── Volume detection & model path wiring ────────────────────────────────────
#
# RunPod mounts network volumes at a path you set in the endpoint config.
# We try several well-known paths, then fall back to the local container path.
# Set RUNPOD_VOLUME_PATH explicitly in the endpoint env vars to override.
#
COMFYUI_ROOT="/comfyui"
LOCAL_MODELS="${COMFYUI_ROOT}/models"

# Candidate volume roots (in priority order)
VOLUME_CANDIDATES=(
    "${RUNPOD_VOLUME_PATH:-}"          # explicit override wins
    "/runpod-volume"                   # RunPod default
    "/workspace"                       # common alternative
)

VOLUME_ROOT=""
for candidate in "${VOLUME_CANDIDATES[@]}"; do
    if [ -n "$candidate" ] && [ -d "$candidate" ]; then
        VOLUME_ROOT="$candidate"
        break
    fi
done

if [ -n "$VOLUME_ROOT" ]; then
    log "Network volume detected : $VOLUME_ROOT"
    MODEL_STORE="${VOLUME_ROOT}/models"

    # Wire ComfyUI's models dir to the volume
    mkdir -p "$MODEL_STORE"
    if [ -L "$LOCAL_MODELS" ]; then
        log "Symlink already exists  : $LOCAL_MODELS → $(readlink $LOCAL_MODELS)"
    elif [ -d "$LOCAL_MODELS" ] && [ ! -L "$LOCAL_MODELS" ]; then
        log "Moving existing local models dir into volume..."
        mv "$LOCAL_MODELS" "${MODEL_STORE}_bak_$(date +%s)" || true
        ln -s "$MODEL_STORE" "$LOCAL_MODELS"
        log "Symlink created         : $LOCAL_MODELS → $MODEL_STORE"
    else
        ln -s "$MODEL_STORE" "$LOCAL_MODELS"
        log "Symlink created         : $LOCAL_MODELS → $MODEL_STORE"
    fi
else
    log "WARNING: No network volume found. Models will be downloaded to ephemeral"
    log "         container storage and LOST when this worker shuts down."
    log "         Set RUNPOD_VOLUME_PATH or mount a volume at /runpod-volume."
    MODEL_STORE="$LOCAL_MODELS"
fi

# ─── Model download guard ─────────────────────────────────────────────────────
#
# We write a manifest file to the volume after a successful download.
# On subsequent starts the manifest check is a single fast stat call.
#
MANIFEST="${MODEL_STORE}/.models_ready"
CHECKPOINT="${MODEL_STORE}/checkpoints/10Eros/10Eros_v1-fp8mixed_learned.safetensors"

sep
if [ -f "$MANIFEST" ]; then
    log "[MODELS] Manifest found  : $MANIFEST"
    log "[MODELS] Written         : $(cat $MANIFEST)"
    # Sanity-check the largest file is actually present and non-empty
    if [ -f "$CHECKPOINT" ] && [ -s "$CHECKPOINT" ]; then
        log "[MODELS] Checkpoint OK   : $CHECKPOINT"
        log "[MODELS] Skipping download."
    else
        log "[MODELS] WARNING: manifest exists but checkpoint missing or empty!"
        log "[MODELS] Re-downloading..."
        rm -f "$MANIFEST"
        HF_TOKEN="${HF_TOKEN:-}" CIVITAI_TOKEN="${CIVITAI_TOKEN:-}" \
            MODEL_STORE="$MODEL_STORE" bash /src/download_models.sh
    fi
else
    log "[MODELS] No manifest — first cold start. Downloading models..."
    log "[MODELS] Destination     : $MODEL_STORE"
    HF_TOKEN="${HF_TOKEN:-}" CIVITAI_TOKEN="${CIVITAI_TOKEN:-}" \
        MODEL_STORE="$MODEL_STORE" bash /src/download_models.sh
    echo "downloaded at $(ts())" > "$MANIFEST"
    log "[MODELS] Manifest written: $MANIFEST"
fi
sep

# ─── extra_model_paths.yaml ───────────────────────────────────────────────────
# Tell ComfyUI to scan the volume's model subdirs. This is belt-and-suspenders:
# the symlink above already works, but this makes it explicit.
cat > "${COMFYUI_ROOT}/extra_model_paths.yaml" <<YAML
ltx_worker:
  base_path: ${MODEL_STORE}
  checkpoints: checkpoints
  loras: loras
  text_encoders: text_encoders
  latent_upscale_models: latent_upscale_models
  vae: vae
YAML
log "[CONFIG] extra_model_paths.yaml written pointing to $MODEL_STORE"

# ─── Launch ComfyUI ───────────────────────────────────────────────────────────
sep
log "[START] Launching ComfyUI..."
log "[START] Args: --listen 127.0.0.1 --port 8188 --disable-auto-launch --disable-metadata"

python /comfyui/main.py \
    --listen 127.0.0.1 \
    --port 8188 \
    --disable-auto-launch \
    --disable-metadata \
    > /tmp/comfyui.log 2>&1 &

COMFY_PID=$!
log "[START] ComfyUI PID: $COMFY_PID"

# ─── Wait for ComfyUI readiness ───────────────────────────────────────────────
log "[WAIT] Waiting for ComfyUI on port 8188 (max 270s)..."
READY=0
for i in $(seq 1 90); do
    ELAPSED=$(( i * 3 ))
    if curl -sf http://127.0.0.1:8188/system_stats > /dev/null 2>&1; then
        log "[READY] ComfyUI is up after ${ELAPSED}s"
        READY=1
        break
    fi

    # Print ComfyUI tail every 15s so RunPod logs show progress
    if (( i % 5 == 0 )); then
        log "[WAIT] ${ELAPSED}s elapsed — ComfyUI log tail:"
        tail -n 5 /tmp/comfyui.log 2>/dev/null | while IFS= read -r line; do
            log "  | $line"
        done
    fi

    # Check ComfyUI hasn't already crashed
    if ! kill -0 $COMFY_PID 2>/dev/null; then
        log "[ERROR] ComfyUI process died unexpectedly. Full log:"
        cat /tmp/comfyui.log | while IFS= read -r line; do log "  | $line"; done
        exit 1
    fi

    sleep 3
done

if [ $READY -eq 0 ]; then
    log "[ERROR] ComfyUI did not become ready in 270s. Full log:"
    cat /tmp/comfyui.log | while IFS= read -r line; do log "  | $line"; done
    exit 1
fi

# Dump the system_stats response for debugging
log "[INFO] ComfyUI system_stats:"
curl -s http://127.0.0.1:8188/system_stats | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    for k,v in d.items():
        print(f'  {k}: {v}')
except Exception as e:
    print(f'  (could not parse: {e})')
" | while IFS= read -r line; do log "$line"; done

sep
log "[START] Launching RunPod handler..."
exec python /handler.py