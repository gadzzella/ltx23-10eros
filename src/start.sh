#!/bin/bash
set -euo pipefail

echo "=== LTX 2.3 / 10Eros Worker Starting ==="

# Start ComfyUI with lowvram flag (required for Gemma + 10Eros to coexist)
echo "[START] Launching ComfyUI..."
python /comfyui/main.py \
    --listen 127.0.0.1 \
    --port 8188 \
    --disable-auto-launch \
    --disable-metadata \
    --lowvram \
    &

# Wait for ComfyUI to be ready
echo "[WAIT] Waiting for ComfyUI on port 8188..."
for i in $(seq 1 90); do
    if curl -sf http://127.0.0.1:8188/system_stats > /dev/null 2>&1; then
        echo "[READY] ComfyUI is up after ${i}s"
        break
    fi
    if [ $i -eq 90 ]; then
        echo "[ERROR] ComfyUI did not start in time."
        exit 1
    fi
    sleep 3
done

echo "[START] Launching RunPod handler..."
python /handler.py
