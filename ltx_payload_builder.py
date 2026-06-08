"""
ltx_payload_builder.py

Injects user-supplied parameters into the LTX 2.3 / 10Eros ComfyUI workflow.
Workflow file: video_ltx23_10eros_i2v_API.json (default, overridable via WORKFLOW_PATH env var)

Node map:
  267:266  PrimitiveStringMultiline  -> positive prompt
  267:247  CLIPTextEncode            -> negative prompt
  269      LoadImage                 -> source image filename
  267:257  PrimitiveInt              -> width
  267:258  PrimitiveInt              -> height
  267:225  PrimitiveInt              -> num_frames
  267:260  PrimitiveInt              -> fps
  267:216  RandomNoise               -> seed (main sampler)
  267:237  RandomNoise               -> seed (refinement sampler)
  267:201  PrimitiveBoolean          -> bypass_i2v
  267:280  LoraLoaderModelOnly       -> Penile Praxis strength
  267:281  LoraLoaderModelOnly       -> Anal Insertion strength
  267:282  LoraLoaderModelOnly       -> DR34ML4Y strength
"""

import json
import os
import random

WORKFLOW_PATH = os.environ.get(
    "WORKFLOW_PATH",
    os.path.join(os.path.dirname(__file__), "video_ltx23_10eros_i2v_API.json")
)

ASPECT_PRESETS = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "1:1":  (768, 768),
    "4:3":  (1024, 768),
    "3:4":  (768, 1024),
}


def _snap_frames(n: int) -> int:
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
    width: int = 1280,
    height: int = 720,
    aspect_ratio: str | None = None,
    num_frames: int = 121,
    fps: int = 25,
    seed: int | None = None,
    bypass_i2v: bool = False,
    lora_penile_strength: float = 0.85,
    lora_anal_strength: float = 0.85,
    lora_dr34ml4y_strength: float = 0.85,
) -> tuple[dict, list[dict]]:
    with open(WORKFLOW_PATH) as f:
        workflow = json.load(f)

    if aspect_ratio and aspect_ratio in ASPECT_PRESETS:
        width, height = ASPECT_PRESETS[aspect_ratio]

    num_frames = _snap_frames(num_frames)

    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    seed2 = (seed + 1) % (2**32)

    workflow["267:266"]["inputs"]["value"] = prompt
    workflow["267:247"]["inputs"]["text"] = negative_prompt
    workflow["267:257"]["inputs"]["value"] = width
    workflow["267:258"]["inputs"]["value"] = height
    workflow["267:225"]["inputs"]["value"] = num_frames
    workflow["267:260"]["inputs"]["value"] = fps
    workflow["267:216"]["inputs"]["noise_seed"] = seed
    workflow["267:237"]["inputs"]["noise_seed"] = seed2
    workflow["267:201"]["inputs"]["value"] = bypass_i2v

    # LoRA strengths
    workflow["267:280"]["inputs"]["strength_model"] = lora_penile_strength
    workflow["267:281"]["inputs"]["strength_model"] = lora_anal_strength
    workflow["267:282"]["inputs"]["strength_model"] = lora_dr34ml4y_strength

    images = []
    if image_b64 and not bypass_i2v:
        fname = image_filename or "source_image.png"
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        workflow["269"]["inputs"]["image"] = fname
        images.append({"name": fname, "image": image_b64})
    elif bypass_i2v:
        workflow["269"]["inputs"]["image"] = "placeholder.png"

    return workflow, images


def seconds_to_frames(seconds: float, fps: int = 25) -> int:
    return _snap_frames(int(round(seconds * fps)))


def frames_to_seconds(frames: int, fps: int = 25) -> float:
    return round(frames / fps, 2)