#!/usr/bin/env python3
"""Label a test recording: for each frame, mark the boxes where someone is really SITTING.

    python3 -m labelling.label recordings/20261011-153440 [d3]      ->  http://localhost:8098
(name cameras after the folder to label only those)

Each frame starts from the model's answer (prelabels) or your earlier labels; you fix it.
Saved after every frame to labels_<cam>.json, which training/train_seat.py and training/daily.py read.
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from seatsense.mistakes import parse
from seatsense.presence import load_boxes

PORT = 8098
HERE = Path(__file__).parent


def validate_labels(labels, boxes, times):
    for t, sitting in labels.items():
        if int(t) not in times:
            raise ValueError(f"no labelling frame at {t} s")
        if not set(sitting) <= boxes:
            raise ValueError(f"unknown box in {sitting}")
    return labels


def notes_for(frame_ts, reports):
    """{frame t: ["You reported at HH:MM:SS: desk N — what"]}: each report marks its nearest frame."""
    out = {}
    for r in reports:
        if frame_ts:
            t = min(frame_ts, key=lambda x: abs(x - r["t"]))
            hms = f"{r['t'] // 3600:02d}:{r['t'] % 3600 // 60:02d}:{r['t'] % 60:02d}"
            out.setdefault(t, []).append(f"You reported at {hms}: desk {r['desk']} — {r['what']}")
    return out


def catalogue(rec, only=()):
    """{cam: {"boxes": [...], "frames": [{"t", "file", "sitting"}]}} for every (or each named) camera."""
    out = {}
    for pre in sorted(rec.glob("prelabels_*.json")):
        cam = pre.stem.split("_", 1)[1]
        if only and cam not in only:
            continue
        start = json.loads(pre.read_text())
        lab = rec / f"labels_{cam}.json"
        done = json.loads(lab.read_text()) if lab.exists() else {}
        frames = sorted(rec.glob(f"label_frames/{cam}_*.jpg"))
        rep = rec / "reported.txt"
        notes = notes_for([int(f.stem.split("_")[1]) for f in frames], parse(rep.read_text()) if rep.exists() else [])
        out[cam] = {"boxes": sorted(load_boxes(rec / f"{cam}_focus.json")),
                    "frames": [{"t": int(f.stem.split("_")[1]), "file": f.name,
                                "labelled": str(int(f.stem.split("_")[1])) in done,
                                "notes": notes.get(int(f.stem.split("_")[1]), []),
                                "sitting": done.get(str(int(f.stem.split("_")[1])),
                                                    start.get(str(int(f.stem.split("_")[1])), []))}
                               for f in frames]}
    return out


def main():
    rec = Path(sys.argv[1]).resolve()
    only = tuple(sys.argv[2:])

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
            if self.path.split("?")[0] == "/":   # "/?at=16:37" opens the page at that time
                return self.send(200, (HERE / "label.html").read_bytes(), "text/html")
            if self.path == "/catalogue":
                return self.send(200, json.dumps(catalogue(rec, only)).encode(), "application/json")
            name = self.path.removeprefix("/img/")
            f = rec / "label_frames" / name
            if self.path.startswith("/img/") and "/" not in name and f.exists():
                return self.send(200, f.read_bytes(), "image/jpeg")
            self.send(404, b"not found", "text/plain")

        def do_POST(self):
            cam = self.path.removeprefix("/labels/")
            cat = catalogue(rec, only)
            if cam not in cat:
                return self.send(404, b"unknown camera", "text/plain")
            try:
                labels = validate_labels(json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
                                         set(cat[cam]["boxes"]), {f["t"] for f in cat[cam]["frames"]})
            except (ValueError, TypeError, AttributeError) as e:
                return self.send(400, str(e).encode(), "text/plain")
            (rec / f"labels_{cam}.json").write_text(json.dumps(labels, indent=1))
            self.send(200, b"saved", "text/plain")

    print(f"labelling {rec.name} ({', '.join(only) or 'all cameras'}) on http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
