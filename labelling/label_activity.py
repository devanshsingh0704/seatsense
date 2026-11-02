#!/usr/bin/env python3
"""Label seated people as working / not working, for the activity model.

    python3 -m labelling.label_activity recordings/activity [--port 8100]   ->  http://localhost:8099 (or the port given)

One crop per seated person (made by training/activity_crops.py, listed in manifest.json).
Keys: W working · P phone · T talking/turned away · I idle/other · U unclear (not trained on).
Saved after every crop to labels.json.
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = 8099
HERE = Path(__file__).parent
ANSWERS = {"W": "working", "P": "not", "T": "not", "I": "not", "U": None}


def validate(labels, files):
    for name, a in labels.items():
        if name not in files:
            raise ValueError(f"no crop {name}")
        if a not in ANSWERS:
            raise ValueError(f"answer must be one of {''.join(ANSWERS)}")
    return labels


def to_state(answer):
    return ANSWERS[answer]


def main():
    root = Path(sys.argv[1]).resolve()
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else PORT   # 8099 may be taken by another project
    manifest = json.loads((root / "manifest.json").read_text())
    files = {m["file"] for m in manifest}
    labels_file = root / "labels.json"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                return self.send(200, (HERE / "label_activity.html").read_bytes(), "text/html")
            if self.path == "/data":
                done = json.loads(labels_file.read_text()) if labels_file.exists() else {}
                return self.send(200, json.dumps({"crops": manifest, "labels": done}).encode(), "application/json")
            name = self.path.removeprefix("/img/")
            if self.path.startswith("/img/") and name in files:
                return self.send(200, (root / "crops" / name).read_bytes(), "image/jpeg")
            self.send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/labels":
                return self.send(404, b"not found", "text/plain")
            try:
                labels = validate(json.loads(self.rfile.read(int(self.headers["Content-Length"]))), files)
            except (ValueError, TypeError, AttributeError) as e:
                return self.send(400, str(e).encode(), "text/plain")
            labels_file.write_text(json.dumps(labels, indent=1))
            self.send(200, b"saved", "text/plain")

    print(f"labelling {len(manifest)} seated people on http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
