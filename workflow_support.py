import json
import time
import uuid
import base64
import urllib.request
import websocket

COMFY_HOST = "127.0.0.1:8188"


def submit_workflow(workflow: dict) -> str:
    payload = json.dumps({
        "prompt": workflow,
        "client_id": str(uuid.uuid4())
    }).encode()
    req = urllib.request.Request(
        f"http://{COMFY_HOST}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())["prompt_id"]


def wait_for_completion(prompt_id: str, timeout: int = 900) -> dict:
    ws = websocket.WebSocket()
    ws.connect(f"ws://{COMFY_HOST}/ws?clientId={prompt_id}")
    deadline = time.time() + timeout
    try:
        while time.time() < deadline:
            raw = ws.recv()
            if not raw:
                continue
            msg = json.loads(raw)
            if msg.get("type") == "executing":
                data = msg.get("data", {})
                if data.get("prompt_id") == prompt_id and data.get("node") is None:
                    break
    finally:
        ws.close()

    with urllib.request.urlopen(
        f"http://{COMFY_HOST}/history/{prompt_id}"
    ) as resp:
        history = json.loads(resp.read())

    return history.get(prompt_id, {})


def get_output_files(history_entry: dict) -> list:
    outputs = []
    for node_id, node_output in history_entry.get("outputs", {}).items():
        for vid in node_output.get("videos", []):
            path = _get_path(vid)
            if path:
                outputs.append(_read_file(path, vid["filename"], "video/mp4"))
        for img in node_output.get("images", []):
            if img.get("type") == "output":
                path = _get_path(img)
                if path:
                    outputs.append(_read_file(path, img["filename"], "image/png"))
    return outputs


def _get_path(item: dict):
    filename = item.get("filename", "")
    subfolder = item.get("subfolder", "")
    if not filename:
        return None
    return f"/comfyui/output/{subfolder}/{filename}" if subfolder else f"/comfyui/output/{filename}"


def _read_file(path: str, filename: str, media_type: str) -> dict:
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return {"filename": filename, "media_type": media_type, "type": "base64", "data": data}
