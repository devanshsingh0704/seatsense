#!/usr/bin/env python3
"""Cut one picture per seated person for working / not-working labelling.

    python3 tools/activity_crops.py recordings/activity recordings/20261012-day recordings/20261017-log

For every desk the labeller marked as sat-in (labels_d20.json), the desk box plus a margin is
cut from the FULL-resolution frame (hands and phones need the pixels), at most 640 px, with
the desk's own box outlined so the labeller judges the right person. Writes <out>/crops/ and
<out>/manifest.json; existing labels in <out>/labels.json are left alone.
"""
import json
import sys
from pathlib import Path

WS = Path(__file__).resolve().parent.parent
MARGIN = 0.15
MAX_SIDE = 640


def cut(img, b):
    """The desk box plus MARGIN from a full-resolution frame, box outlined, at most MAX_SIDE."""
    import cv2
    sys.path.insert(0, str(WS / "tools"))
    from crops import crop_box
    x1, y1, x2, y2 = crop_box(b, img.shape[1], img.shape[0], MARGIN)
    c = img[y1:y2, x1:x2].copy()
    cv2.rectangle(c, (b[0] - x1, b[1] - y1), (b[2] - x1, b[3] - y1), (0, 255, 0), 3)
    s = MAX_SIDE / max(c.shape[:2])
    return cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else c


def main():
    import cv2
    sys.path.insert(0, str(WS)); sys.path.insert(0, str(WS / "tools"))
    from presence import load_boxes
    out = Path(sys.argv[1]); (out / "crops").mkdir(parents=True, exist_ok=True)
    manifest = []
    for day in map(Path, sys.argv[2:]):
        boxes = load_boxes(day / "d20_focus.json")
        labels = json.loads((day / "labels_d20.json").read_text())
        for t in sorted(labels, key=int):
            if not labels[t]:
                continue
            img = cv2.imread(str(day / "frames" / f"d20_{int(t):05d}.jpg"))
            for n in labels[t]:
                name = f"{day.name}_{int(t):05d}_{n}.jpg"
                cv2.imwrite(str(out / "crops" / name), cut(img, boxes[n]), [cv2.IMWRITE_JPEG_QUALITY, 90])
                manifest.append({"file": name, "dir": day.name, "t": int(t), "box": n})
        print(f"{day.name}: {sum(m['dir'] == day.name for m in manifest)} seated people", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest))
    print(f"total {len(manifest)}")


if __name__ == "__main__":
    main()
