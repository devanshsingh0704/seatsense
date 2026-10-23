#!/usr/bin/env python3
"""Turn a day of hourly recordings into labelling frames for the "seat occupied?" classifier.

    python3 tools/day_frames.py recordings/20261012-day

For every <cam>_HHMMSS.mkv, one frame per full clock minute (files are read start to end,
because a file cut off by a restart has no index to seek in). Writes, per frame:
  frames/<cam>_<second-of-day>.jpg         full resolution -- desk-box crops for training come from these
  label_frames/<cam>_<second-of-day>.jpg   the same frame with the focus boxes drawn, for labelling
  prelabels_<cam>.json                     boxes the current rules count as sat in, as a starting point
  <cam>_focus.json                         the boxes in force (copied, so later edits can't skew the labels)
tools/label.py works on the result unchanged.
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

WS = Path(__file__).resolve().parent.parent
EVERY = 60
W, H = 2560, 1440


def segment_start(name):
    hh, mm, ss = re.search(r"_(\d\d)(\d\d)(\d\d)\.mkv$", name).groups()
    return int(hh) * 3600 + int(mm) * 60 + int(ss)


def is_sample(start, k, every=EVERY):
    """Frame k of a 1-frame-per-second read of a file that started at `start` (second of day)."""
    return (start + k) % every == 0


def main():
    import cv2
    import numpy as np
    from ultralytics import YOLO
    sys.path.insert(0, str(WS))
    sys.path.insert(0, str(WS / "tools"))
    from extract import draw_boxes          # same box drawing as the first labelling round
    from presence import assign, load_boxes
    from vest import is_visitor

    day = Path(sys.argv[1])
    (day / "frames").mkdir(exist_ok=True)
    (day / "label_frames").mkdir(exist_ok=True)
    model = YOLO(str(WS / "models" / "yolo26m.pt"))
    cams = sorted({p.name.split("_")[0] for p in day.glob("*.mkv")})
    for cam in cams:
        shutil.copy(WS / "zones" / f"{cam}_focus.json", day / f"{cam}_focus.json")
        boxes = load_boxes(day / f"{cam}_focus.json")
        pre = {}
        for video in sorted(day.glob(f"{cam}_*.mkv")):
            start = segment_start(video.name)
            proc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", str(video), "-an", "-vf", "fps=1",
                                     "-pix_fmt", "bgr24", "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
            k = 0
            while len(buf := proc.stdout.read(W * H * 3)) == W * H * 3:
                if is_sample(start, k):
                    t = start + k
                    img = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
                    cv2.imwrite(str(day / "frames" / f"{cam}_{t:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    small = cv2.resize(img, (1280, 720))
                    cv2.imwrite(str(day / "label_frames" / f"{cam}_{t:05d}.jpg"), draw_boxes(small, boxes),
                                [cv2.IMWRITE_JPEG_QUALITY, 85])
                    r = model.predict(small, classes=[0], conf=0.35, imgsz=960, half=True, verbose=False)[0]
                    people = [b for b in r.boxes.xyxy.tolist() if not is_visitor(small, b)]
                    owners = assign([[v * 2 for v in b] for b in people], boxes)
                    pre[str(t)] = sorted(n for n, idx in owners.items() if idx)
                k += 1
            proc.wait()
            print(f"{video.name}: {k} s read, frames so far {len(pre)}", flush=True)
        (day / f"prelabels_{cam}.json").write_text(json.dumps(pre))
        print(f"{cam}: {len(pre)} labelling frames", flush=True)


if __name__ == "__main__":
    main()
