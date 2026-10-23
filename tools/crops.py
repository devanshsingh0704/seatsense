#!/usr/bin/env python3
"""Cut person crops from a test recording, for training a sitting/standing classifier.

    python3 tools/crops.py recordings/20261011-153440        (CPU only, no GPU)

Uses the detections extract.py already saved, one frame every EVERY seconds, and keeps
people at least partly inside a desk box (the current zones/). Each crop gets a margin so
the chair and desk show, and a first guess -- "sit" if the person fits their box >= 85%,
else "stand" -- for the labeller to confirm or fix. Writes crops/<cam>_<second>_<i>.jpg
and crops/manifest.json.
"""
import json
import subprocess
import sys
from pathlib import Path

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS))
from presence import MIN_INSIDE, _inside_fraction, load_boxes   # noqa: E402

EVERY = 20          # seconds between sampled frames: people sit still, closer frames repeat
W, H = 2560, 1440
MAX_SIDE = 320      # crops are stored at most this big


def pick_people(persons, boxes, min_conf=0.35, min_inside=0.3):
    out = []
    for p in persons:
        frac, n = max((_inside_fraction(p[:4], b), n) for n, b in boxes.items())
        if p[4] >= min_conf and frac >= min_inside:
            out.append({"bbox": p[:4], "conf": p[4], "box": n, "inside": round(frac, 2),
                        "guess": "sit" if frac >= MIN_INSIDE else "stand"})
    return out


def crop_box(b, w, h, margin=0.2):
    x1, y1, x2, y2 = b
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    return (max(0, round(x1 - mx)), max(0, round(y1 - my)), min(w, round(x2 + mx)), min(h, round(y2 + my)))


def main():
    import cv2
    import numpy as np

    rec = Path(sys.argv[1])
    out_dir = rec / "crops"
    out_dir.mkdir(exist_ok=True)
    manifest = []
    for det in sorted(rec.glob("detections_*.jsonl")):
        cam = det.stem.split("_", 1)[1]
        boxes = load_boxes(WS / "zones" / f"{cam}_focus.json")
        frames = [json.loads(l) for l in det.read_text().splitlines()]
        n_cam = 0
        times = range(0, len(frames), EVERY)
        for t in times:   # seek to each sampled second instead of decoding the whole video
            buf = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", str(t), "-i", str(rec / f"{cam}.mkv"),
                                  "-frames:v", "1", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"],
                                 capture_output=True).stdout
            if len(buf) != W * H * 3:
                continue
            img = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            for i, p in enumerate(pick_people(frames[t]["persons"], boxes)):
                x1, y1, x2, y2 = crop_box(p["bbox"], W, H)
                crop = img[y1:y2, x1:x2]
                s = MAX_SIDE / max(crop.shape[:2])
                if s < 1:
                    crop = cv2.resize(crop, (round(crop.shape[1] * s), round(crop.shape[0] * s)))
                name = f"{cam}_{t:05d}_{i}.jpg"
                cv2.imwrite(str(out_dir / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                manifest.append({"file": name, "cam": cam, "t": t, **p})
                n_cam += 1
        print(f"{cam}: {n_cam} crops from {len(times)} frames", flush=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    guesses = [m["guess"] for m in manifest]
    print(f"total {len(manifest)} crops: guess sit {guesses.count('sit')}, stand {guesses.count('stand')}")


if __name__ == "__main__":
    main()
