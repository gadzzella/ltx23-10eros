"""
ltx_payload_builder.py

Injects user-supplied parameters into the LTX 2.3 / 10Eros ComfyUI workflow.
Workflow file: video_ltx23_10eros_i2v_API.json (default, overridable via WORKFLOW_PATH env var)

Node map (verified against video_ltx23_10eros_i2v_API.json):
  267:266  PrimitiveStringMultiline    → positive prompt
  267:247  CLIPTextEncode              → negative prompt
  269      LoadImage                   → source image filename
  267:257  PrimitiveInt                → width  (feeds ResizeImageMaskNode + EmptyLTXVLatentVideo via /2)
  267:258  PrimitiveInt                → height (feeds ResizeImageMaskNode + EmptyLTXVLatentVideo via /2)
  267:225  PrimitiveInt                → num_frames
  267:260  PrimitiveInt                → fps
  267:216  RandomNoise                 → seed (refinement sampler — second pass)
  267:237  RandomNoise                 → seed (main sampler — first pass)
  267:201  PrimitiveBoolean            → bypass_i2v
  267:280  LoraLoaderModelOnly         → Penile Praxis strength
  267:281  LoraLoaderModelOnly         → Anal Insertion strength
  267:282  LoraLoaderModelOnly         → DR34ML4Y strength
  267:274  TextGenerateLTX2Prompt      → temperature, top_k, top_p, repetition_penalty, thinking
  267:230  LTXVImgToVideoInplace       → upscaler bypass (latent input swapped)

Width/height wiring confirmed:
  267:257 (width)  → ResizeImageMaskNode.resize_type.width
                   → ComfyMathExpression(a/2)[1] → EmptyLTXVLatentVideo.width
  267:258 (height) → ResizeImageMaskNode.resize_type.height
                   → ComfyMathExpression(a/2)[1] → EmptyLTXVLatentVideo.height

Supported aspect ratios:
  9:16  → 720×1280  (portrait — workflow default)
  16:9  → 1280×720  (landscape)
  1:1   → 1280×1280 (square)
  4:3   → 1024×768
  3:4   → 768×1024
"""

import json
import os
import random

WORKFLOW_PATH = os.environ.get(
    "WORKFLOW_PATH",
    os.path.join(os.path.dirname(__file__), "video_ltx23_10eros_i2v_API.json")
)

# All dimensions are multiples of 8 (required by LTX latent space)
# Default is 9:16 portrait (matches workflow JSON defaults: 267:257=720, 267:258=1280)
ASPECT_PRESETS = {
    "9:16": (720,  1280),   # portrait — workflow default
    "16:9": (1280,  720),   # landscape
    "1:1":  (1280, 1280),   # square
    "4:3":  (1024,  768),
    "3:4":  (768,  1024),
}


def _snap_frames(n: int) -> int:
    """
    LTX requires frame count to satisfy: (n - 1) % 8 == 0
    Snap to nearest valid value, minimum 25 frames.
    """
    if n < 25:
        n = 25
    remainder = (n - 1) % 8
    if remainder == 0:
        return n
    lower = n - remainder
    upper = lower + 8
    return lower if (n - lower) <= (upper - n) else upper


def build_payload(
    prompt: str,
    image_b64: str | None = None,
    image_filename: str | None = None,
    negative_prompt: str = "pc game, console game, video game, cartoon, childish, ugly",
    width: int = 720,
    height: int = 1280,
    aspect_ratio: str | None = None,
    num_frames: int = 121,
    fps: int = 25,
    seed: int | None = None,
    bypass_i2v: bool = False,
    lora_penile_strength: float = 0.85,
    lora_anal_strength: float = 0.85,
    lora_dr34ml4y_strength: float = 0.85,
    upscaler_enabled: bool = True,
    temperature: float = 0.7,
    top_k: int = 64,
    top_p: float = 0.95,
    repetition_penalty: float = 1.05,
    thinking: bool = False,
) -> tuple[dict, list[dict]]:
    with open(WORKFLOW_PATH) as f:
        workflow = json.load(f)

    # Aspect ratio preset overrides explicit width/height
    if aspect_ratio and aspect_ratio in ASPECT_PRESETS:
        width, height = ASPECT_PRESETS[aspect_ratio]

    num_frames = _snap_frames(num_frames)

    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    # Second sampler gets seed+1 (wraps at 2^32) for deterministic variation
    seed2 = (seed + 1) % (2**32)

    # ── Core parameters ───────────────────────────────────────────────────────
    workflow["267:266"]["inputs"]["value"] = prompt
    workflow["267:247"]["inputs"]["text"]  = negative_prompt

    # Width → node 267:257, Height → node 267:258
    # (verified: 267:257 feeds ResizeImageMaskNode.width and EmptyLTXVLatentVideo.width via /2)
    workflow["267:257"]["inputs"]["value"] = width
    workflow["267:258"]["inputs"]["value"] = height

    workflow["267:225"]["inputs"]["value"] = num_frames
    workflow["267:260"]["inputs"]["value"] = fps

    # 267:237 = main sampler (first pass), 267:216 = refinement sampler (second pass)
    workflow["267:237"]["inputs"]["noise_seed"] = seed
    workflow["267:216"]["inputs"]["noise_seed"] = seed2

    workflow["267:201"]["inputs"]["value"] = bypass_i2v

    # ── LoRA strengths ────────────────────────────────────────────────────────
    workflow["267:280"]["inputs"]["strength_model"] = lora_penile_strength
    workflow["267:281"]["inputs"]["strength_model"] = lora_anal_strength
    workflow["267:282"]["inputs"]["strength_model"] = lora_dr34ml4y_strength

    # ── LLM prompt enhancer (TextGenerateLTX2Prompt) ─────────────────────────
    workflow["267:274"]["inputs"]["sampling_mode.temperature"]        = temperature
    workflow["267:274"]["inputs"]["sampling_mode.top_k"]              = top_k
    workflow["267:274"]["inputs"]["sampling_mode.top_p"]              = top_p
    workflow["267:274"]["inputs"]["sampling_mode.repetition_penalty"] = repetition_penalty
    workflow["267:274"]["inputs"]["thinking"]                         = thinking
    workflow["267:274"]["inputs"]["sampling_mode.seed"]               = seed

    # ── Upscaler wiring ───────────────────────────────────────────────────────
    # upscaler_enabled=True:  267:230 latent comes from 267:253 (LTXVLatentUpsampler)
    # upscaler_enabled=False: 267:230 latent comes from 267:217 (LTXVSeparateAVLatent, bypasses upscaler)
    if upscaler_enabled:
        workflow["267:230"]["inputs"]["latent"] = ["267:253", 0]
    else:
        workflow["267:230"]["inputs"]["latent"] = ["267:217", 0]

    # ── Input image ───────────────────────────────────────────────────────────
    images = []
    if image_b64 and not bypass_i2v:
        fname = image_filename or "source_image.png"
        # Strip data URI prefix if present
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        workflow["269"]["inputs"]["image"] = fname
        images.append({"name": fname, "image": image_b64})
    else:
        # T2V mode (bypass_i2v=True) or no image supplied:
        # placeholder.png is baked at /comfyui/input/placeholder.png during Docker build.
        # LoadImage is always executed in the graph (used by TextGenerateLTX2Prompt
        # and ResizeImageMaskNode), so it must point to a file that exists.
        workflow["269"]["inputs"]["image"] = "placeholder.png"

    return workflow, images


def seconds_to_frames(seconds: float, fps: int = 25) -> int:
    return _snap_frames(int(round(seconds * fps)))


def frames_to_seconds(frames: int, fps: int = 25) -> float:
    return round(frames / fps, 2)
