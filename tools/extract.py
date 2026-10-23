#!/usr/bin/env python3
"""Run the detector over a test recording once and save what it saw. Runs on the GPU laptop.

    python3 tools/extract.py recordings/20261011-153440

For each <cam>.mkv, at 1 frame per second, writes:
  detections_<cam>.jsonl   every person box (source pixels + confidence), for score.py
  label_frames/<cam>_<second>.jpg   one frame a minute with the focus boxes, for labelling
  prelabels_<cam>.json     what the current rules said at those seconds (labelling starts from it)
  <cam>_focus.json         a copy of the boxes in force, so later re-marking can't skew the score
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS))
sys.path.insert(0, str(WS / "tools"))
from presence import load_boxes   # noqa: E402
from score import replay          # noqa: E402

W, H, SRC_W = 1280, 720, 2560
IMGSZ, CONF = 960, 0.35            # same as live.py
LABEL_EVERY = 60                   # seconds between labelling frames


def frames_1fps(video):
    cmd = ["ffmpeg", "-loglevel", "error", "-i", str(video), "-an", "-vf", f"fps=1,scale={W}:{H}",
           "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    while len(buf := proc.stdout.read(W * H * 3)) == W * H * 3:
        yield np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    proc.wait()


def draw_boxes(frame, boxes):
    out = frame.copy()
    for n, b in boxes.items():
        x1, y1, x2, y2 = (round(v * W / SRC_W) for v in b)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(str(n), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(out, (x1 + 4, y1 + 4), (x1 + 12 + tw, y1 + 12 + th), (0, 0, 0), -1)
        cv2.putText(out, str(n), (x1 + 8, y1 + 8 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    rec = Path(sys.argv[1])
    model = YOLO(str(WS / "models" / "yolo26m.pt"))
    (rec / "label_frames").mkdir(exist_ok=True)
    for video in sorted(rec.glob("*.mkv")):
        cam = video.stem
        shutil.copy(WS / "zones" / f"{cam}_focus.json", rec / f"{cam}_focus.json")
        boxes = load_boxes(rec / f"{cam}_focus.json")
        frames = []
        with open(rec / f"detections_{cam}.jsonl", "w") as out:
            for t, img in enumerate(frames_1fps(video)):
                r = model.predict(img, classes=[0], conf=CONF, imgsz=IMGSZ, half=True, verbose=False)[0]
                persons = [[round(v * SRC_W / W) for v in b] + [round(c, 3)]
                           for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]
                frames.append({"t": t, "persons": persons})
                out.write(json.dumps(frames[-1]) + "\n")
                if t % LABEL_EVERY == 0 and t > 0:
                    cv2.imwrite(str(rec / "label_frames" / f"{cam}_{t:05d}.jpg"), draw_boxes(img, boxes),
                                [cv2.IMWRITE_JPEG_QUALITY, 85])
        states = replay(frames, boxes)
        pre = {t: [n for n, s in states[t].items() if s == "occupied"]
               for t in states if t % LABEL_EVERY == 0 and t > 0}
        (rec / f"prelabels_{cam}.json").write_text(json.dumps(pre))
        print(f"{cam}: {len(frames)} s processed, {len(pre)} labelling frames", flush=True)


if __name__ == "__main__":
    main()
