# ltx23-10eros

RunPod serverless worker for LTX 2.3 / 10Eros image-to-video and text-to-video with synchronized audio.

## Stack

- ComfyUI (latest master)
- ComfyUI-LTXVideo (Lightricks official nodes)
- ComfyMath (ComfyMathExpression)
- `10Eros_v1-fp8mixed_learned.safetensors` — main checkpoint (video + audio VAE bundled)
- `Gemma_3_12B_it_fp8_scaled.safetensors` — text encoder
- `ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors` — distilled speed LoRA
- `Penile_Praxis_V4.safetensors` — NSFW LoRA
- `nsfw_anal_insertion_ltx23_v1.0.safetensors` — NSFW LoRA
- `DR34ML4Y_LTXXX_V2.safetensors` — NSFW LoRA
- `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` — latent upscaler

## Repo structure

```
.
├── Dockerfile
├── requirements.txt
├── handler.py
├── ltx_payload_builder.py
├── workflow_support.py
├── video_ltx23_10eros_i2v_API.json
├── frontend/
│   └── index.html
└── src/
    ├── start.sh
    └── download_models.sh
```

## Setup

### GitHub Secrets (Actions)

| Secret | Notes |
|--------|-------|
| `DOCKERHUB_TOKEN` | Docker Hub access token |
| `HF_TOKEN` | HuggingFace token |
| `CIVITAI_TOKEN` | Civitai API token |

### RunPod Endpoint Config

- **Image:** `gadzela/ltx23-10eros:latest`
- **GPU:** 80GB (A100 or H100)
- **Container disk:** 80GB minimum

## API

### Input

```json
{
  "input": {
    "prompt": "A woman walks through a corridor...",
    "image": "<base64>",
    "image_filename": "source.png",
    "negative_prompt": "cartoon, ugly",
    "aspect_ratio": "9:16",
    "seconds": 5,
    "fps": 25,
    "seed": 42,
    "bypass_i2v": false,
    "lora_penile_strength": 0.85,
    "lora_anal_strength": 0.85,
    "lora_dr34ml4y_strength": 0.85
  }
}
```

### Output

```json
{
  "output": [
    {
      "filename": "LTX_2.3_i2v_00001_.mp4",
      "media_type": "video/mp4",
      "type": "base64",
      "data": "<base64 mp4>"
    }
  ]
}
```

## Frontend

Open `frontend/index.html` directly in a browser. No server needed. Enter your RunPod endpoint URL and API key, upload an image (I2V) or skip (T2V), set LoRA strengths, and generate.
