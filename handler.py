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
import logging
import time
import runpod
from ltx_payload_builder import build_payload, seconds_to_frames
from workflow_support import submit_workflow, wait_for_completion, get_output_files

# ─── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    force=True,
)
log = logging.getLogger("handler")


def handler(job: dict) -> dict:
    job_id = job.get("id", "unknown")
    t0 = time.time()
    log.info("═" * 60)
    log.info("JOB START  id=%s", job_id)
    log.debug("Raw job keys: %s", list(job.keys()))

    inp = job.get("input", {})
    log.debug("Input fields: %s", {k: (v if k != "image" else f"<base64 {len(v or '')} chars>")
                                    for k, v in inp.items()})

    # ── Validation ────────────────────────────────────────────────────────────
    prompt = inp.get("prompt", "").strip()
    if not prompt:
        log.error("Missing required field: prompt")
        return {"error": "Missing required field: prompt"}

    bypass_i2v = bool(inp.get("bypass_i2v", False))
    image_b64 = inp.get("image")

    if not bypass_i2v and not image_b64:
        log.error("I2V mode requires 'image' field. bypass_i2v=%s", bypass_i2v)
        return {"error": "I2V mode requires 'image' (base64). Set bypass_i2v=true for T2V."}

    log.info("Mode: %s", "T2V (bypass_i2v)" if bypass_i2v else "I2V")
    log.info("Prompt length: %d chars", len(prompt))

    # ── Parameter parsing ─────────────────────────────────────────────────────
    fps = int(inp.get("fps", 25))
    if "seconds" in inp:
        num_frames = seconds_to_frames(float(inp["seconds"]), fps)
        log.info("Duration: %.1fs → %d frames @ %d fps", float(inp["seconds"]), num_frames, fps)
    else:
        num_frames = int(inp.get("num_frames", 121))
        log.info("Frames: %d @ %d fps", num_frames, fps)

    seed = int(inp["seed"]) if "seed" in inp else None
    log.info("Seed: %s", seed if seed is not None else "random")

    aspect_ratio = inp.get("aspect_ratio")
    width  = int(inp.get("width", 1280))
    height = int(inp.get("height", 720))
    log.info("Resolution: %dx%d  aspect_ratio=%s", width, height, aspect_ratio or "none")

    lora_penile   = float(inp.get("lora_penile_strength", 0.85))
    lora_anal     = float(inp.get("lora_anal_strength", 0.85))
    lora_dr34ml4y = float(inp.get("lora_dr34ml4y_strength", 0.85))
    upscaler      = bool(inp.get("upscaler_enabled", True))
    log.info("LoRAs: penile=%.2f  anal=%.2f  dr34ml4y=%.2f  upscaler=%s",
             lora_penile, lora_anal, lora_dr34ml4y, upscaler)

    # ── Build workflow payload ────────────────────────────────────────────────
    log.info("Building workflow payload...")
    t_build = time.time()
    try:
        workflow, images = build_payload(
            prompt=prompt,
            image_b64=image_b64,
            image_filename=inp.get("image_filename", "source_image.png"),
            negative_prompt=inp.get("negative_prompt",
                                    "pc game, console game, video game, cartoon, childish, ugly"),
            width=width,
            height=height,
            aspect_ratio=aspect_ratio,
            num_frames=num_frames,
            fps=fps,
            seed=seed,
            bypass_i2v=bypass_i2v,
            lora_penile_strength=lora_penile,
            lora_anal_strength=lora_anal,
            lora_dr34ml4y_strength=lora_dr34ml4y,
            upscaler_enabled=upscaler,
            temperature=float(inp.get("temperature", 0.7)),
            top_k=int(inp.get("top_k", 64)),
            top_p=float(inp.get("top_p", 0.95)),
            repetition_penalty=float(inp.get("repetition_penalty", 1.05)),
            thinking=bool(inp.get("thinking", False)),
        )
    except Exception as exc:
        log.exception("Failed to build workflow payload: %s", exc)
        return {"error": f"Payload build failed: {exc}"}

    log.info("Payload built in %.2fs  nodes=%d  images=%d",
             time.time() - t_build, len(workflow), len(images))

    # ── Write input images ────────────────────────────────────────────────────
    if images:
        input_dir = "/comfyui/input"
        os.makedirs(input_dir, exist_ok=True)
        for img in images:
            dest = os.path.join(input_dir, img["name"])
            try:
                img_bytes = base64.b64decode(img["image"])
                with open(dest, "wb") as f:
                    f.write(img_bytes)
                log.info("Image written: %s (%d bytes)", dest, len(img_bytes))
            except Exception as exc:
                log.exception("Failed to write image %s: %s", img["name"], exc)
                return {"error": f"Failed to write input image '{img['name']}': {exc}"}

    # ── Submit workflow ───────────────────────────────────────────────────────
    log.info("Submitting workflow to ComfyUI...")
    t_submit = time.time()
    try:
        prompt_id = submit_workflow(workflow)
    except Exception as exc:
        log.exception("Failed to submit workflow: %s", exc)
        return {"error": f"ComfyUI submission failed: {exc}"}
    log.info("Workflow submitted in %.2fs  prompt_id=%s", time.time() - t_submit, prompt_id)

    # ── Wait for completion ───────────────────────────────────────────────────
    timeout = int(inp.get("timeout", 900))
    log.info("Waiting for completion  prompt_id=%s  timeout=%ds", prompt_id, timeout)
    t_exec = time.time()
    try:
        history = wait_for_completion(prompt_id, timeout=timeout)
    except Exception as exc:
        log.exception("Error while waiting for completion: %s", exc)
        return {"error": f"Workflow execution error: {exc}"}

    elapsed_exec = time.time() - t_exec
    if not history:
        log.error("Workflow returned no history after %.1fs  prompt_id=%s", elapsed_exec, prompt_id)
        return {"error": f"Workflow timed out or returned no history. prompt_id={prompt_id}"}

    log.info("Workflow completed in %.1fs", elapsed_exec)
    log.debug("History keys: %s", list(history.keys()))

    # ── Collect output files ──────────────────────────────────────────────────
    log.info("Collecting output files...")
    try:
        outputs = get_output_files(history)
    except Exception as exc:
        log.exception("Failed to read output files: %s", exc)
        return {"error": f"Output file read failed: {exc}"}

    if not outputs:
        log.error("Workflow completed but produced no output files. History: %s", history)
        return {"error": "Workflow completed but produced no output files."}

    for o in outputs:
        log.info("Output: %s  type=%s  media_type=%s  data_len=%d",
                 o.get("filename"), o.get("type"), o.get("media_type"),
                 len(o.get("data", "")))

    total = time.time() - t0
    log.info("JOB DONE  id=%s  outputs=%d  total_time=%.1fs", job_id, len(outputs), total)
    log.info("═" * 60)

    return {"output": outputs}


runpod.serverless.start({"handler": handler})
