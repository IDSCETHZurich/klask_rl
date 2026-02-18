"""Reusable HTTP image server.

Streams a single numpy RGB image to the browser with auto-refresh.

Usage:
    from mjpeg_server import MJPEGServer

    server = MJPEGServer(port=8089)
    server.start()
    server.update_frame(rgb_array)   # (H, W, 3) uint8
    server.stop()
"""

import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

import cv2
import numpy as np


class _ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class MJPEGServer:
    """Threaded HTTP server that serves a JPEG snapshot with auto-refresh."""

    def __init__(self, port: int = 8089, title: str = "Camera Debug", quality: int = 90):
        self.port = port
        self.title = title
        self.quality = quality
        self._lock = threading.Lock()
        self._jpeg: bytes = b""
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self):
        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/" or self.path.startswith("/?"):
                    self._serve_page()
                elif self.path.startswith("/frame"):
                    self._serve_frame()
                else:
                    self.send_error(404)

            def _serve_page(self):
                html = f"""\
<!DOCTYPE html>
<html>
<head>
  <title>{server_ref.title}</title>
  <style>
    body {{ margin:0; background:#111; display:flex; align-items:center;
           justify-content:center; height:100vh; }}
    #feed {{ max-width:100%; max-height:100vh; image-rendering:pixelated; }}
  </style>
</head>
<body>
  <img id="feed"/>
  <script>
    setInterval(() => {{
      const img = document.getElementById("feed");
      const tmp = new Image();
      tmp.onload = () => {{ img.src = tmp.src; }};
      tmp.src = "/frame?t=" + Date.now();
    }}, 100);
  </script>
</body>
</html>"""
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(html.encode())

            def _serve_frame(self):
                with server_ref._lock:
                    jpeg = server_ref._jpeg
                if not jpeg:
                    self.send_error(204, "No frame")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.send_header("Cache-Control", "no-cache, no-store")
                self.end_headers()
                self.wfile.write(jpeg)

            def log_message(self, fmt, *args):
                pass

        self._server = _ThreadedHTTPServer(("0.0.0.0", self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def update_frame(self, img: np.ndarray, resize: tuple[int, int] | None = (320, 420)):
        """Push a new RGB frame (H, W, 3) uint8."""
        if resize is not None:
            img = cv2.resize(img, resize, interpolation=cv2.INTER_NEAREST)
        _, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        with self._lock:
            self._jpeg = buf.tobytes()

    def print_url(self):
        print(f"[INFO] View stream at http://localhost:{self.port}")

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
