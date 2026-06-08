"""
handler.py — RunPod serverless handler for LTX 2.3 / 10Eros

Input fields:
  prompt                 str    required
  image                  str    optional  base64 image (required for I2V)
  image_filename         str    optional  default: source_image.png
  negative_prompt        str    optional
  aspect_ratio           str    optional  16:9 | 9:16 | 1:1 | 4:3 | 3:4
  width                  int    optional  default 1280 (overridden by aspect_ratio)
  height                 int    optional  default 720  (overridden by aspect_ratio)
  seconds                float  optional  duration — overrides num_frames
  num_frames             int    optional  default 121
  fps                    int    optional  default 25
  seed                   int    optional  random if omitted
  bypass_i2v             bool   optional  True=T2V, False=I2V (default False)
  lora_penile_strength   float  optional  default 0.85
  lora_anal_strength     float  optional  default 0.85
  lora_dr34ml4y_strength float  optional  default 0.85
  upscaler_enabled       bool   optional  default True
  temperature            float  optional  LLM sampling temperature (default 0.7)
  top_k                  int    optional  LLM top-k (default 64)
  top_p                  float  optional  LLM top-p (default 0.95)
  repetition_penalty     float  optional  LLM repetition penalty (default 1.05)
  thinking               bool   optional  LLM thinking mode (default False)
"""

import os
import base64
import runpod
from ltx_payload_builder import build_payload, seconds_to_frames
from workflow_support import submit_workflow, wait_for_completion, get_output_files


def handler(job: dict) -> dict:
    inp = job.get("input", {})

    prompt = inp.get("prompt", "").strip()
    if not prompt:
        return {"error": "Missing required field: prompt"}

    bypass_i2v = bool(inp.get("bypass_i2v", False))
    image_b64 = inp.get("image")

    if not bypass_i2v and not image_b64:
        return {"error": "I2V mode requires 'image' (base64). Set bypass_i2v=true for T2V."}

    fps = int(inp.get("fps", 25))
    if "seconds" in inp:
        num_frames = seconds_to_frames(float(inp["seconds"]), fps)
    else:
        num_frames = int(inp.get("num_frames", 121))

    seed = int(inp["seed"]) if "seed" in inp else None

    workflow, images = build_payload(
        prompt=prompt,
        image_b64=image_b64,
        image_filename=inp.get("image_filename", "source_image.png"),
        negative_prompt=inp.get("negative_prompt", "pc game, console game, video game, cartoon, childish, ugly"),
        width=int(inp.get("width", 1280)),
        height=int(inp.get("height", 720)),
        aspect_ratio=inp.get("aspect_ratio"),
        num_frames=num_frames,
        fps=fps,
        seed=seed,
        bypass_i2v=bypass_i2v,
        lora_penile_strength=float(inp.get("lora_penile_strength", 0.85)),
        lora_anal_strength=float(inp.get("lora_anal_strength", 0.85)),
        lora_dr34ml4y_strength=float(inp.get("lora_dr34ml4y_strength", 0.85)),
        upscaler_enabled=bool(inp.get("upscaler_enabled", True)),
        temperature=float(inp.get("temperature", 0.7)),
        top_k=int(inp.get("top_k", 64)),
        top_p=float(inp.get("top_p", 0.95)),
        repetition_penalty=float(inp.get("repetition_penalty", 1.05)),
        thinking=bool(inp.get("thinking", False)),
    )

    # Write images to ComfyUI input folder
    if images:
        input_dir = "/comfyui/input"
        os.makedirs(input_dir, exist_ok=True)
        for img in images:
            img_bytes = base64.b64decode(img["image"])
            with open(os.path.join(input_dir, img["name"]), "wb") as f:
                f.write(img_bytes)

    prompt_id = submit_workflow(workflow)
    history = wait_for_completion(prompt_id, timeout=900)

    if not history:
        return {"error": f"Workflow timed out or returned no history. prompt_id={prompt_id}"}

    outputs = get_output_files(history)
    if not outputs:
        return {"error": "Workflow completed but produced no output files."}

    return {"output": outputs}


runpod.serverless.start({"handler": handler})