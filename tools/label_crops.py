#!/usr/bin/env python3
"""Label person crops as sitting / standing, for the posture classifier.

    python3 tools/label_crops.py recordings/20261011-153440      ->  http://localhost:8099

Each crop starts from the guess in crops/manifest.json (or your earlier answer).
Keys: S = sitting, T = standing, U = unclear (left out of training). Saved after every
crop to crops/labels.json.
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = 8099
HERE = Path(__file__).parent
ANSWERS = {"sit", "stand", "unclear"}


def validate_crop_labels(labels, files):
    for name, answer in labels.items():
        if name not in files:
            raise ValueError(f"no crop {name}")
        if answer not in ANSWERS:
            raise ValueError(f"answer must be one of {sorted(ANSWERS)}")
    return labels


def main():
    crops = Path(sys.argv[1]).resolve() / "crops"
    manifest = json.loads((crops / "manifest.json").read_text())
    files = {m["file"] for m in manifest}
    labels_file = crops / "labels.json"

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
                return self.send(200, (HERE / "label_crops.html").read_bytes(), "text/html")
            if self.path == "/data":
                done = json.loads(labels_file.read_text()) if labels_file.exists() else {}
                return self.send(200, json.dumps({"crops": manifest, "labels": done}).encode(), "application/json")
            name = self.path.removeprefix("/img/")
            if self.path.startswith("/img/") and name in files:
                return self.send(200, (crops / name).read_bytes(), "image/jpeg")
            self.send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/labels":
                return self.send(404, b"not found", "text/plain")
            try:
                labels = validate_crop_labels(json.loads(self.rfile.read(int(self.headers["Content-Length"]))), files)
            except (ValueError, TypeError, AttributeError) as e:
                return self.send(400, str(e).encode(), "text/plain")
            labels_file.write_text(json.dumps(labels, indent=1))
            self.send(200, b"saved", "text/plain")

    print(f"labelling {len(manifest)} crops on http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
