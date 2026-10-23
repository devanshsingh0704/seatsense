#!/usr/bin/env python3
"""Mark the focus boxes for one camera on a fresh frame from the NVR.

    ~/desk_presence_ws/.venv/bin/python ~/desk_presence_ws/tools/mark_boxes.py d3

Drag a box, press ENTER, drag the next, ENTER... then ESC when done (in the image window).
Each box is saved the moment it is confirmed, so Ctrl+C or closing the window loses nothing.
Saves zones/<cam>_focus.json as {"source_size": [W, H], "regions": [[x1, y1, x2, y2], ...]}
in full-resolution pixels; the previous file is kept as zones/<cam>_focus.prev.json.
The frame you marked on is saved as snapshots/<cam>.jpg.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

WS = Path(__file__).resolve().parent.parent
CHANNELS = {"d20": 20}   # D3 dropped 2026-10-16
MAX_SHOW_W = 1600   # the 2560-wide frame is shown scaled to fit the screen


def to_regions(boxes, scale):
    """selectROIs (x, y, w, h) in window pixels -> [x1, y1, x2, y2] in source pixels."""
    out = []
    for x, y, w, h in boxes:
        r = [round(x / scale), round(y / scale), round((x + w) / scale), round((y + h) / scale)]
        if w >= 8 and h >= 8 and r not in out:   # skip stray clicks and an ENTER pressed twice
            out.append(r)
    return out


def grab_frame(cam):
    env = dict(l.split("=", 1) for l in (WS / ".env").read_text().split() if "=" in l)
    url = (f"rtsp://{env['NVR_USER']}:{env['NVR_PASS']}@{env['NVR_HOST']}:554"
           f"/cam/realmonitor?channel={CHANNELS[cam]}&subtype=0")
    out = WS / "snapshots" / f"{cam}.jpg"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-rtsp_transport", "tcp", "-i", url,
                    "-frames:v", "1", "-q:v", "2", str(out)], check=True, timeout=30)
    return out


def main():
    import cv2   # only needed for the window; keeps to_regions testable without OpenCV

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    add = "--add" in sys.argv   # keep the existing boxes and append new ones after them
    cam = args[0] if args else ""
    if cam not in CHANNELS:
        print(f"usage: mark_boxes.py {{{'|'.join(CHANNELS)}}} [full-resolution frame.jpg] [--add]")
        return 1
    if len(args) > 1:   # mark on a frame with people seated, so boxes include their heads
        frame = cv2.imread(args[1])
    else:
        print(f"grabbing a fresh frame from {cam}...")
        frame = cv2.imread(str(grab_frame(cam)))
    H, W = frame.shape[:2]
    scale = min(1.0, MAX_SHOW_W / W)
    shown = cv2.resize(frame, (round(W * scale), round(H * scale))) if scale < 1 else frame

    target = WS / "zones" / f"{cam}_focus.json"
    existing = json.loads(target.read_text())["regions"] if add and target.exists() else []
    if existing:
        print(f"adding to the {len(existing)} existing box(es); new boxes are numbered from {len(existing) + 1}")
    # when redrawing, show the old boxes faintly so each desk keeps its number (labels depend on it)
    reference = json.loads(target.read_text())["regions"] if not add and target.exists() else []
    if reference:
        print(f"redrawing: the {len(reference)} old boxes are shown dark grey -- draw box 1 over old 1, 2 over old 2, ...")
    window = f"mark focus boxes - {cam}  (ENTER = keep box, ESC = finished)"
    print("in the IMAGE window: drag a box, press ENTER; repeat; press ESC when finished.")
    print("every box is saved the moment you press ENTER, so closing early loses nothing.")
    boxes = []
    try:
        while True:
            canvas = shown.copy()   # keep earlier boxes visible, numbered
            for i, (x1, y1, x2, y2) in enumerate(reference, 1):  # old boxes, for reference only
                p1, p2 = (round(x1 * scale), round(y1 * scale)), (round(x2 * scale), round(y2 * scale))
                cv2.rectangle(canvas, p1, p2, (90, 90, 90), 1)
                cv2.putText(canvas, f"old {i}", (p1[0] + 6, p2[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 90), 2)
            for i, (x1, y1, x2, y2) in enumerate(existing, 1):   # existing boxes in grey
                p1, p2 = (round(x1 * scale), round(y1 * scale)), (round(x2 * scale), round(y2 * scale))
                cv2.rectangle(canvas, p1, p2, (170, 170, 170), 2)
                cv2.putText(canvas, str(i), (p1[0] + 8, p1[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (170, 170, 170), 2)
            for i, (x, y, w, h) in enumerate(boxes, len(existing) + 1):
                cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(canvas, str(i), (x + 8, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            x, y, w, h = map(int, cv2.selectROI(window, canvas, False))
            if w == 0 or h == 0:    # ESC or c with nothing drawn = finished
                break
            boxes.append((x, y, w, h))
            regions = existing + to_regions(boxes, scale)
            if len(boxes) == 1 and target.exists():   # back up the old boxes once, before the first save
                os.replace(target, target.with_name(f"{cam}_focus.prev.json"))
            target.write_text(json.dumps({"source_size": [W, H], "regions": regions}, indent=2))
            print(f"  box {len(regions)} saved")
    except KeyboardInterrupt:
        print("\nstopped with Ctrl+C")
    cv2.destroyAllWindows()
    for _ in range(4):
        cv2.waitKey(1)
    if not boxes:
        print("nothing marked; the existing file is unchanged")
        return 1
    regions = existing + to_regions(boxes, scale)
    print(f"{len(regions)} box(es) in {target}")
    for i, (x1, y1, x2, y2) in enumerate(regions, 1):
        print(f"  {i}: {x1},{y1} -> {x2},{y2}  ({x2 - x1}x{y2 - y1})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
