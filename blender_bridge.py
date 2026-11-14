# blender_bridge.py — isophote <-> Blender HTTP bridge.
# Tiny stdlib-only server the web app and the painterly_normals addon talk to.
# Endpoints (all POST, JSON, CORS enabled for the Pages origin):
#   /poll          -> {kind: "none"} or {kind: "image", bgr_png: base64}
#   /paint         body {image_png: base64, params{...}} -> {png: base64}
#   /send_to_blender body {png: base64} -> {ok}
#   /status        -> {status: "..."}
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 8778
_lock = threading.Lock()
_state = {
    # pending image the app should paint next (from Blender)
    "pending_bgr_png": None,
    # last painted result the app pushed back
    "last_result_png": None,
    # human-readable app status line
    "status": "",
}


def _publish(event, **payload):
    """Called by painterly_normals: drop a PNG (bytes or b64 str) into the bridge."""
    with _lock:
        if event == "from_blender":
            _state["pending_bgr_png"] = payload["png_b64"]
        elif event == "poll":
            pass
        elif event == "result":
            _state["last_result_png"] = payload["png_b64"]


def _status(msg: str):
    with _lock:
        _state["status"] = msg


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quiet
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _reply(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._reply({"error": "bad json"}, 400)

        if self.path == "/queue":
            # Blender (addon) pushes the next image for the web app to paint
            with _lock:
                _state["pending_bgr_png"] = req.get("png")
            return self._reply({"ok": True})

        if self.path == "/poll":
            with _lock:
                png = _state.get("pending_bgr_png")
                _state["pending_bgr_png"] = None
                status = _state["status"]
            return self._reply({"kind": "image", "bgr_png": png} if png
                               else {"kind": "none", "status": status})

        if self.path == "/paint":
            return self._reply(_paint(req))

        if self.path == "/send_to_blender":
            with _lock:
                _state["last_result_png"] = req.get("png")
            return self._reply({"ok": True})

        if self.path == "/status":
            with _lock:
                return self._reply({"status": _state["status"],
                                    "has_result": _state.get("last_result_png") is not None,
                                    "result_len": len(_state.get("last_result_png") or "")})

        return self._reply({"error": "unknown path"}, 404)


def _paint(req):
    """Decode the app's PNG payload, paint it with the isophote core, return b64 PNG."""
    try:
        src = base64.b64decode(req["image_png"])
        params = req.get("params") or {}
        import numpy as np
        from io import BytesIO
        from PIL import Image
        import cv2  # noqa: F401  (isophote imports it anyway)
        im = Image.open(BytesIO(src)).convert("RGBA")
        arr = np.asarray(im)
        # RGBA -> BGRA order the python core expects
        bgra = arr[..., [2, 1, 0, 3]].copy()
        from isophote import paint_bytes
        png_b, secs = paint_bytes(
            bgra, bgra.shape[0], bgra.shape[1],
            params.get("brush_len", 26), params.get("brush_w", 7),
            params.get("layers", 2), params.get("relief", 0.35),
            params.get("base_radius", 6), params.get("flow_smooth", 8),
            params.get("seed", 42))
        b64 = base64.b64encode(png_b).decode()
        _status(f"painted in {secs:.2f}s")
        return {"png": b64, "secs": round(secs, 2)}
    except Exception as e:
        return {"error": str(e)}


_server = None


def start(port=PORT):
    """Start (idempotent) the bridge server on a daemon thread."""
    global _server
    if _server is not None:
        return
    _server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    t = threading.Thread(target=_server.serve_forever, daemon=True)
    t.start()
