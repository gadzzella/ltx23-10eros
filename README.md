# ltx23-10eros

RunPod serverless worker for **LTX 2.3 / 10Eros** — image-to-video and text-to-video with synchronized audio.

All models are **baked into the Docker image at build time**. No network volume is required. Cold start is ~60–90s.

---

## Stack

| Component | Details |
|-----------|---------|
| ComfyUI | Latest master via `comfy-cli` |
| ComfyUI-LTXVideo | Lightricks official nodes (all LTXV/audio nodes) |
| ComfyUI-KJNodes | `ResizeImageMaskNode`, `ResizeImagesByLongerEdge` |
| ComfyUI-VideoHelperSuite | `CreateVideo`, `SaveVideo` |
| ComfyUI-Manager | `PreviewAny` and node management |
| ComfyMath | `ComfyMathExpression` (width/2, height/2 for latent sizing) |
| `10Eros_v1-fp8mixed_learned.safetensors` | Main checkpoint — video VAE + audio VAE bundled (~30 GB) |
| `Gemma_3_12B_it_fp8_scaled.safetensors` | Text encoder (~13 GB) |
| `ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors` | Distilled speed LoRA (~1 GB) |
| `Penile_Praxis_V4.safetensors` | NSFW LoRA |
| `nsfw_anal_insertion_ltx23_v1.0.safetensors` | NSFW LoRA |
| `DR34ML4Y_LTXXX_V2.safetensors` | NSFW LoRA |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | Latent spatial upscaler (~950 MB) |

---

## Repo structure

```
.
├── Dockerfile
├── extra_model_paths.yaml
├── requirements.txt
├── handler.py
├── ltx_payload_builder.py
├── workflow_support.py
├── video_ltx23_10eros_i2v_API.json
├── frontend/
│   └── index.html
└── src/
    └── start.sh
```

> `download_models.sh` has been removed — models are baked at build time, not downloaded at runtime.

---

## Building the image

**Use RunPod's Build & Push tool.** Do NOT use GitHub Actions — the runner disk (~14 GB free) is far too small for a ~75 GB baked image.

### Step-by-step

1. Go to **RunPod → Storage → Build & Push** (or the equivalent in your RunPod dashboard).
2. Connect your GitHub repo (`gadzzella/ltx23-10eros`).
3. Set the following **Build Secrets** (treated like `--build-arg`, not stored in the final image):

   | Secret name | Value |
   |-------------|-------|
   | `HF_TOKEN` | Your HuggingFace token (needed for gated `TenStrip` repos) |
   | `CIVITAI_TOKEN` | Your Civitai API token |

4. Set **Docker Hub credentials**:
   - Username: `gadzela`
   - Token: your Docker Hub access token
   - Image tag: `gadzela/ltx23-10eros:latest`

5. Select **A40 48 GB** GPU for the build job (sufficient disk + fast network).

6. Start the build. The build will:
   - Install all custom nodes
   - Download all 7 model files (~46 GB total) as separate cached Docker layers
   - Print a final inventory confirming every file size
   - Push the finished image (~75–80 GB compressed) to Docker Hub

> **Layer caching:** each model is a separate `RUN wget` layer. If you rebuild after changing only code files, Docker reuses the cached model layers and the rebuild completes in minutes.

---

## RunPod Endpoint configuration

| Setting | Value |
|---------|-------|
| **Container image** | `gadzela/ltx23-10eros:latest` |
| **GPU** | 80 GB (A100 SXM or H100 SXM) |
| **Container disk** | 100 GB minimum |
| **Network volume** | Not required (models are baked in) |
| **Environment variables** | None required at runtime |

**Do not attach a network volume.** The models are inside the image. A volume would not be used and wastes money.

---

## API

### Input fields

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `prompt` | str | **required** | |
| `image` | str | — | Base64 image. Required for I2V mode. |
| `image_filename` | str | `source_image.png` | |
| `negative_prompt` | str | `"pc game, console game, video game, cartoon, childish, ugly"` | |
| `aspect_ratio` | str | — | `9:16` / `16:9` / `1:1` / `4:3` / `3:4`. Overrides width/height. |
| `width` | int | `720` | Ignored if `aspect_ratio` is set. |
| `height` | int | `1280` | Ignored if `aspect_ratio` is set. |
| `seconds` | float | — | Video duration. Overrides `num_frames`. |
| `num_frames` | int | `121` | Used if `seconds` not set. Snapped to `(n-1) % 8 == 0`. |
| `fps` | int | `25` | |
| `seed` | int | random | |
| `bypass_i2v` | bool | `false` | `true` = T2V mode (no input image needed). |
| `lora_penile_strength` | float | `0.85` | |
| `lora_anal_strength` | float | `0.85` | |
| `lora_dr34ml4y_strength` | float | `0.85` | |
| `upscaler_enabled` | bool | `true` | `false` skips the spatial upscaler pass. |
| `temperature` | float | `0.7` | TextGenerateLTX2Prompt LLM sampling temperature. |
| `top_k` | int | `64` | |
| `top_p` | float | `0.95` | |
| `repetition_penalty` | float | `1.05` | |
| `thinking` | bool | `false` | LLM thinking mode. |
| `timeout` | int | `900` | Max seconds to wait for workflow completion. |

### Aspect ratio reference

| `aspect_ratio` | Resolution | Use case |
|---------------|------------|---------|
| `9:16` | 720 × 1280 | Portrait / mobile (workflow default) |
| `16:9` | 1280 × 720 | Landscape / widescreen |
| `1:1` | 1280 × 1280 | Square |
| `4:3` | 1024 × 768 | |
| `3:4` | 768 × 1024 | |

### Example request (I2V)

```json
{
  "input": {
    "prompt": "A woman walks through a corridor, cinematic lighting, photorealistic",
    "image": "<base64>",
    "aspect_ratio": "9:16",
    "seconds": 5,
    "fps": 25,
    "seed": 42
  }
}
```

### Example request (T2V)

```json
{
  "input": {
    "prompt": "A woman stands by a window in warm light, cinematic, photorealistic",
    "bypass_i2v": true,
    "aspect_ratio": "16:9",
    "seconds": 4
  }
}
```

### Output

```json
{
  "output": [
    {
      "filename": "video/LTX_2.3_i2v_00001_.mp4",
      "media_type": "video/mp4",
      "type": "base64",
      "data": "<base64 mp4>"
    }
  ]
}
```

---

## Node map (workflow cross-reference)

| Node ID | Class | Purpose |
|---------|-------|---------|
| `267:201` | PrimitiveBoolean | `bypass_i2v` flag |
| `267:221` | LTXVAudioVAELoader | Loads audio VAE from **checkpoints/** folder |
| `267:236` | CheckpointLoaderSimple | Loads main checkpoint |
| `267:243` | LTXAVTextEncoderLoader | Loads Gemma text encoder + checkpoint |
| `267:232` | LoraLoaderModelOnly | Distilled LoRA (strength 0.5, fixed) |
| `267:280` | LoraLoaderModelOnly | Penile Praxis LoRA (strength = API input) |
| `267:281` | LoraLoaderModelOnly | Anal Insertion LoRA (strength = API input) |
| `267:282` | LoraLoaderModelOnly | DR34ML4Y LoRA (strength = API input) |
| `267:233` | LatentUpscaleModelLoader | Spatial upscaler |
| `267:238` | ResizeImageMaskNode | Image resize to target W×H (KJNodes) |
| `267:235` | ResizeImagesByLongerEdge | Pre-upscale resize to 1536 long edge (KJNodes) |
| `267:242` | CreateVideo | Combine frames + audio (VideoHelperSuite) |
| `75` | SaveVideo | Save output video (VideoHelperSuite) |
| `267:257` | PrimitiveInt | **Width** (feeds image resize + latent W/2) |
| `267:258` | PrimitiveInt | **Height** (feeds image resize + latent H/2) |
| `267:274` | TextGenerateLTX2Prompt | LLM prompt enhancement |

> **Important:** `LTXVAudioVAELoader` (node `267:221`) loads its model from the `checkpoints/` folder, not `vae/`. The 10Eros checkpoint bundles the audio VAE inside the same `.safetensors` file.

---

## Frontend

Open `frontend/index.html` directly in a browser. No server needed. Enter your RunPod endpoint ID and API key, upload an image (I2V) or skip (T2V), set parameters, and generate.
