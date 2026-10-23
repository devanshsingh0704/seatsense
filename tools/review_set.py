#!/usr/bin/env python3
"""Pick the minutes worth labelling from a frame-logger day and prepare them for tools/label.py.

    python3 tools/review_set.py recordings/20261017-log [HH:MM ...]      # HH:MM = times the user reported

Picked: any desk unsure (score between 0.05 and 0.95), desk 5 occupied (its known false
alarm), neighbouring desks (1+4, 1+5) occupied together, every third such minute, the reported times
(plus two minutes either side), and
every EVERY-th remaining minute as a confident control. Writes label_frames/ (boxes drawn),
prelabels_d20.json (the live answer, as a starting point) and d20_focus.json beside frames/.
"""
import json
import shutil
import sys
from pathlib import Path

WS = Path(__file__).resolve().parent.parent
UNSURE = (0.05, 0.95)
EVERY = 10
# desks whose boxes overlap in the picture: someone at one can sit inside the other's box
NEIGHBOURS = (("1", "4"), ("1", "5"))
PAIR_EVERY = 3


def select(rows, reported, every=EVERY):
    picked = set()
    for r in rows:
        d = r["desks"]
        occ = {n for n, (s, _) in d.items() if s != "vacant"}
        if any(p is not None and UNSURE[0] < p < UNSURE[1] for _, p in d.values()) or "5" in occ:
            picked.add(r["t"])
    pairs = [r["t"] for r in rows if r["t"] not in picked and any(
        r["desks"].get(a, ["vacant"])[0] != "vacant" and r["desks"].get(b, ["vacant"])[0] != "vacant" for a, b in NEIGHBOURS)]
    picked.update(pairs[::PAIR_EVERY])
    for t in reported:                       # the reported minute and two either side
        for d in (-120, -60, 0, 60, 120):
            picked.add(min((r["t"] for r in rows), key=lambda x: abs(x - (t + d))))
    rest = [r["t"] for r in rows if r["t"] not in picked]
    picked.update(rest[::every])
    return sorted(picked)


def logged(day):
    """The live answers for the minutes that have a frame."""
    rows = [json.loads(l) for l in (day / "live_d20.jsonl").read_text().splitlines() if l.strip()]
    have = {int(f.stem.split("_")[1]) for f in (day / "frames").glob("d20_*.jpg")}
    return sorted((r for r in rows if r["t"] in have), key=lambda r: r["t"])


def build(day, reported, extra=()):
    """Write the review set for `day`: the picked minutes plus `extra` ones. Returns them."""
    import cv2
    sys.path.insert(0, str(WS)); sys.path.insert(0, str(WS / "tools"))
    from extract import draw_boxes
    from presence import load_boxes
    rows = logged(day)
    ts = sorted(set(select(rows, reported)) | set(extra))
    shutil.copy(WS / "zones" / "d20_focus.json", day / "d20_focus.json")
    boxes = load_boxes(day / "d20_focus.json")
    (day / "label_frames").mkdir(exist_ok=True)
    by_t = {r["t"]: r for r in rows}
    pre = {}
    for t in ts:
        img = cv2.resize(cv2.imread(str(day / "frames" / f"d20_{t:05d}.jpg")), (1280, 720))
        cv2.imwrite(str(day / "label_frames" / f"d20_{t:05d}.jpg"), draw_boxes(img, boxes), [cv2.IMWRITE_JPEG_QUALITY, 85])
        pre[str(t)] = sorted(int(n) for n, (s, _) in by_t[t]["desks"].items() if s != "vacant")
    (day / "prelabels_d20.json").write_text(json.dumps(pre))
    print(f"{len(rows)} logged minutes -> {len(ts)} picked for review")
    return ts


def main():
    day = Path(sys.argv[1])
    build(day, [int(h) * 3600 + int(m) * 60 for h, m in (a.split(":") for a in sys.argv[2:])])


if __name__ == "__main__":
    main()
