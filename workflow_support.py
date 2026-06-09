"""
workflow_support.py — ComfyUI HTTP/WS helpers for the LTX 2.3 / 10Eros worker.

Logs every node execution event so RunPod logs show live workflow progress.
"""

import json
import logging
import time
import uuid
import base64
import urllib.request
import urllib.error
import websocket

log = logging.getLogger("workflow_support")

COMFY_HOST = "127.0.0.1:8188"


# ─── Submit ───────────────────────────────────────────────────────────────────

def submit_workflow(workflow: dict) -> str:
    client_id = str(uuid.uuid4())
    payload = json.dumps({"prompt": workflow, "client_id": client_id}).encode()

    log.debug("Submitting to http://%s/prompt  client_id=%s  payload_size=%d",
              COMFY_HOST, client_id, len(payload))

    req = urllib.request.Request(
        f"http://{COMFY_HOST}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read()
            result = json.loads(body)
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        log.error("ComfyUI /prompt HTTP %d: %s", e.code, error_body)
        raise RuntimeError(f"ComfyUI /prompt returned HTTP {e.code}: {error_body}") from e
    except Exception as e:
        log.error("ComfyUI /prompt request failed: %s", e)
        raise

    # ComfyUI returns validation errors in the body even on 200
    if "error" in result:
        log.error("ComfyUI validation error: %s", result["error"])
        node_errors = result.get("node_errors", {})
        if node_errors:
            for node_id, node_err in node_errors.items():
                log.error("  node %s: %s", node_id, node_err)
        raise RuntimeError(f"ComfyUI workflow validation failed: {result['error']}")

    prompt_id = result.get("prompt_id")
    if not prompt_id:
        log.error("ComfyUI /prompt response missing prompt_id: %s", result)
        raise RuntimeError(f"ComfyUI /prompt response missing prompt_id: {result}")

    log.info("Workflow queued  prompt_id=%s", prompt_id)
    return prompt_id


# ─── Wait for completion ──────────────────────────────────────────────────────

def wait_for_completion(prompt_id: str, timeout: int = 900) -> dict:
    ws_url = f"ws://{COMFY_HOST}/ws?clientId={prompt_id}"
    log.info("Connecting WebSocket  url=%s", ws_url)

    ws = websocket.WebSocket()
    try:
        ws.connect(ws_url, timeout=30)
    except Exception as e:
        log.error("WebSocket connect failed: %s", e)
        raise

    deadline = time.time() + timeout
    last_node = None
    node_start_times: dict[str, float] = {}

    try:
        while time.time() < deadline:
            remaining = deadline - time.time()
            ws.settimeout(min(remaining, 10))

            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                log.debug("WS recv timeout — still waiting (%.0fs left)", remaining)
                continue
            except Exception as e:
                log.error("WS recv error: %s", e)
                raise

            if not raw:
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                log.debug("WS non-JSON frame (%d bytes)", len(raw))
                continue

            msg_type = msg.get("type", "")
            data = msg.get("data", {})

            if msg_type == "status":
                queue = data.get("status", {}).get("exec_info", {}).get("queue_remaining")
                if queue is not None:
                    log.debug("Queue remaining: %d", queue)

            elif msg_type == "execution_start":
                if data.get("prompt_id") == prompt_id:
                    log.info("Execution started  prompt_id=%s", prompt_id)

            elif msg_type == "execution_cached":
                nodes = data.get("nodes", [])
                if nodes:
                    log.info("Cached nodes (skipped): %s", nodes)

            elif msg_type == "executing":
                node = data.get("node")
                pid  = data.get("prompt_id")

                if pid != prompt_id:
                    continue

                if node is None:
                    # None node = workflow finished
                    log.info("Execution finished (node=None signal)  prompt_id=%s", prompt_id)
                    break

                # Log node completion timing
                if last_node and last_node in node_start_times:
                    elapsed = time.time() - node_start_times[last_node]
                    log.info("Node %-12s done in %.1fs", last_node, elapsed)

                node_start_times[node] = time.time()
                log.info("Executing node: %s", node)
                last_node = node

            elif msg_type == "progress":
                value = data.get("value", 0)
                maximum = data.get("max", 1)
                node = data.get("node", "?")
                pct = int(100 * value / max(maximum, 1))
                log.info("Progress  node=%-8s  %3d%%  (%d/%d)", node, pct, value, maximum)

            elif msg_type == "execution_error":
                if data.get("prompt_id") == prompt_id:
                    node_id   = data.get("node_id", "?")
                    node_type = data.get("node_type", "?")
                    exc_msg   = data.get("exception_message", "")
                    exc_type  = data.get("exception_type", "")
                    traceback = data.get("traceback", [])
                    log.error("Execution error on node %s (%s): %s — %s",
                              node_id, node_type, exc_type, exc_msg)
                    for line in traceback:
                        log.error("  %s", line.rstrip())
                    raise RuntimeError(
                        f"ComfyUI node {node_id} ({node_type}) failed: "
                        f"{exc_type}: {exc_msg}"
                    )

            else:
                log.debug("WS msg type=%s  data_keys=%s", msg_type, list(data.keys()))

        else:
            log.error("Timed out waiting for workflow after %ds  prompt_id=%s", timeout, prompt_id)
            raise TimeoutError(f"Workflow did not complete within {timeout}s (prompt_id={prompt_id})")

    finally:
        ws.close()
        log.debug("WebSocket closed")

    # ── Fetch history ──────────────────────────────────────────────────────────
    history_url = f"http://{COMFY_HOST}/history/{prompt_id}"
    log.info("Fetching history  url=%s", history_url)
    try:
        with urllib.request.urlopen(history_url, timeout=30) as resp:
            history = json.loads(resp.read())
    except Exception as e:
        log.error("Failed to fetch history: %s", e)
        raise

    entry = history.get(prompt_id, {})
    log.debug("History entry keys: %s", list(entry.keys()))

    status = entry.get("status", {})
    log.info("History status: completed=%s  messages=%s",
             status.get("completed"), status.get("messages"))

    output_node_count = len(entry.get("outputs", {}))
    log.info("Output nodes in history: %d", output_node_count)

    return entry


# ─── Collect outputs ──────────────────────────────────────────────────────────

def get_output_files(history_entry: dict) -> list:
    outputs = []
    all_outputs = history_entry.get("outputs", {})
    log.debug("Scanning %d output node(s): %s", len(all_outputs), list(all_outputs.keys()))

    for node_id, node_output in all_outputs.items():
        videos = node_output.get("videos", [])
        images = node_output.get("images", [])
        log.debug("Node %s: videos=%d  images=%d", node_id, len(videos), len(images))

        for vid in videos:
            path = _get_path(vid)
            log.info("Video output: node=%s  filename=%s  path=%s",
                     node_id, vid.get("filename"), path)
            if path:
                try:
                    outputs.append(_read_file(path, vid["filename"], "video/mp4"))
                    log.info("Video file read OK: %s", path)
                except Exception as e:
                    log.error("Failed to read video %s: %s", path, e)
                    raise

        for img in images:
            if img.get("type") == "output":
                path = _get_path(img)
                log.info("Image output: node=%s  filename=%s  path=%s",
                         node_id, img.get("filename"), path)
                if path:
                    try:
                        outputs.append(_read_file(path, img["filename"], "image/png"))
                        log.info("Image file read OK: %s", path)
                    except Exception as e:
                        log.error("Failed to read image %s: %s", path, e)
                        raise

    log.info("Total output files collected: %d", len(outputs))
    return outputs


def _get_path(item: dict):
    filename = item.get("filename", "")
    subfolder = item.get("subfolder", "")
    if not filename:
        log.warning("Output item has no filename: %s", item)
        return None
    path = f"/comfyui/output/{subfolder}/{filename}" if subfolder else f"/comfyui/output/{filename}"
    return path


def _read_file(path: str, filename: str, media_type: str) -> dict:
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(f"Output file not found: {path}")
    file_size = os.path.getsize(path)
    log.debug("Reading %s (%d bytes)  media_type=%s", path, file_size, media_type)
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return {"filename": filename, "media_type": media_type, "type": "base64", "data": data}