#!/usr/bin/env python3
"""Run the pose model over a folder of frames once; save every person's box and keypoints.

    python3 tools/pose_extract.py recordings/20261012-day/frames recordings/20261012-day/pose_d20.json

Output: {"<second-of-day>": [{"box": [x1,y1,x2,y2], "conf": c, "kp": [[x, y, conf] * 17]}, ...]}
in full-frame pixels. COCO keypoints: 5/6 shoulders, 11/12 hips, 13/14 knees, 15/16 ankles.
"""
import json
import sys
from pathlib import Path

from ultralytics import YOLO

WS = Path(__file__).resolve().parent.parent


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    model = YOLO(str(WS / "models" / "yolo26m-pose.pt"))
    files = sorted(src.glob("d20_*.jpg"))
    res = {}
    for i in range(0, len(files), 16):
        batch = files[i:i + 16]
        for f, r in zip(batch, model.predict([str(f) for f in batch], conf=0.3, imgsz=1280, half=True, verbose=False)):
            kp = r.keypoints.data.tolist() if r.keypoints is not None else []
            res[str(int(f.stem.split("_")[1]))] = [
                {"box": [round(v) for v in b], "conf": round(c, 3), "kp": [[round(x), round(y), round(k, 3)] for x, y, k in p]}
                for b, c, p in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), kp)]
    out.write_text(json.dumps(res))
    print(f"{len(res)} frames, {sum(len(v) for v in res.values())} people -> {out}")


if __name__ == "__main__":
    main()
