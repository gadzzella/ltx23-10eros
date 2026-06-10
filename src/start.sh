#!/bin/bash
set -uo pipefail
# NOTE: -e is intentionally omitted. Errors are checked explicitly so a single
# failed pipe does not kill the script silently.

# ─── Logging ──────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%dT%H:%M:%S')] $*"; }
sep() { log "═══════════════════════════════════════════════════════════════"; }

sep
log "=== LTX 2.3 / 10Eros Worker Starting (BAKED IMAGE) ==="
log "Hostname      : $(hostname)"
log "CUDA devices  : ${CUDA_VISIBLE_DEVICES:-not set}"

log "NVIDIA SMI:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null \
    | while IFS= read -r line; do log "  GPU: $line"; done || log "  (nvidia-smi unavailable)"

sep

# ─── Verify baked models are present ──────────────────────────────────────────
# This is a fast sanity check only — models are baked at build time, not downloaded here.
log "[MODELS] Verifying baked model files..."

MODELS_OK=1
REQUIRED_FILES=(
    "/comfyui/models/checkpoints/10Eros/10Eros_v1-fp8mixed_learned.safetensors"
    "/comfyui/models/text_encoders/split_files/text_encoders/Gemma_3_12B_it_fp8_scaled.safetensors"
    "/comfyui/models/loras/ltxv/ltx2/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors"
    "/comfyui/models/loras/ltxv/penile_praxis/Penile_Praxis_V4.safetensors"
    "/comfyui/models/loras/ltxv/anal_insertion/nsfw_anal_insertion_ltx23_v1.0.safetensors"
    "/comfyui/models/loras/ltxv/dr34ml4y/DR34ML4Y_LTXXX_V2.safetensors"
    "/comfyui/models/latent_upscale_models/LTX-Video/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
)

for f in "${REQUIRED_FILES[@]}"; do
    if [ -f "$f" ] && [ -s "$f" ]; then
        size=$(du -sh "$f" | cut -f1)
        log "[MODELS] OK   $size  $f"
    else
        log "[MODELS] MISSING OR EMPTY: $f"
        MODELS_OK=0
    fi
done

if [ $MODELS_OK -eq 0 ]; then
    log "[MODELS] ERROR: One or more required model files are missing."
    log "[MODELS] This image was not built correctly. Rebuild with valid HF_TOKEN and CIVITAI_TOKEN."
    exit 1
fi

log "[MODELS] All models verified."
sep

# ─── Verify custom nodes are present ──────────────────────────────────────────
log "[NODES] Verifying custom node directories..."

NODES_OK=1
REQUIRED_NODES=(
    "/comfyui/custom_nodes/ComfyUI-LTXVideo"
    "/comfyui/custom_nodes/comfyui-manager"
    "/comfyui/custom_nodes/ComfyMath"
    "/comfyui/custom_nodes/ComfyUI-KJNodes"
    "/comfyui/custom_nodes/ComfyUI-VideoHelperSuite"
)

for d in "${REQUIRED_NODES[@]}"; do
    if [ -d "$d" ]; then
        log "[NODES] OK   $d"
    else
        log "[NODES] MISSING: $d"
        NODES_OK=0
    fi
done

if [ $NODES_OK -eq 0 ]; then
    log "[NODES] ERROR: One or more custom node directories are missing."
    exit 1
fi

log "[NODES] All custom nodes verified."
sep

# ─── Launch ComfyUI ───────────────────────────────────────────────────────────
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
# Baked image: no model download on startup. ComfyUI should be ready in ~30–60s.
# Allow 270s (4.5 min) to cover cold GPU init and model scanning.
log "[WAIT] Waiting for ComfyUI on port 8188 (max 270s)..."
READY=0
for i in $(seq 1 90); do
    ELAPSED=$(( i * 3 ))

    if curl -sf http://127.0.0.1:8188/system_stats > /dev/null 2>&1; then
        log "[READY] ComfyUI is up after ${ELAPSED}s"
        READY=1
        break
    fi

    # Every 15s print the last 5 lines of ComfyUI log
    if [ $(( i % 5 )) -eq 0 ]; then
        log "[WAIT] ${ELAPSED}s elapsed — ComfyUI log tail:"
        tail -n 5 /tmp/comfyui.log 2>/dev/null | while IFS= read -r line; do
            log "  | $line"
        done
    fi

    # Check ComfyUI process is still alive
    if ! kill -0 $COMFY_PID 2>/dev/null; then
        log "[ERROR] ComfyUI process died (PID $COMFY_PID). Full log:"
        cat /tmp/comfyui.log | while IFS= read -r line; do log "  | $line"; done
        exit 1
    fi

    sleep 3
done

if [ $READY -eq 0 ]; then
    log "[ERROR] ComfyUI did not become ready within 270s. Full log:"
    cat /tmp/comfyui.log | while IFS= read -r line; do log "  | $line"; done
    exit 1
fi

# ─── Print system_stats for debugging ─────────────────────────────────────────
log "[INFO] ComfyUI system_stats:"
curl -s http://127.0.0.1:8188/system_stats | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    for k, v in d.items():
        print(f'  {k}: {v}')
except Exception as e:
    print(f'  (could not parse response: {e})')
" | while IFS= read -r line; do log "$line"; done

sep
log "[START] Launching RunPod handler..."
exec python /handler.py
