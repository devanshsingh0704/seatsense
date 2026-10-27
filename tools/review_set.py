#!/usr/bin/env python3
"""Pick the minutes worth labelling from a frame-logger day and prepare them for tools/label.py.

    python3 tools/review_set.py recordings/20261017-log [HH:MM ...]      # HH:MM = times the user reported

Picked: the minutes where a desk is unsure (calibrated score between 0.05 and 0.95; the MAX_UNSURE
closest to 50/50), desk 5 occupied (its known false
alarm), neighbouring desks (1+4, 1+5) occupied together, every third such minute, the reported times
(plus two minutes either side), and
every EVERY-th remaining minute as a confident control. tools/daily.py caps the set at a daily budget
(see select()), because an uncapped full day was ~290 frames and the labelling stopped. Writes label_frames/ (boxes drawn),
prelabels_d20.json (the live answer, as a starting point) and d20_focus.json beside frames/.
"""
import json
import shutil
import sys
from pathlib import Path

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS / "tools"))
from evaluate import calibrated, load_temperatures   # noqa: E402
UNSURE = (0.05, 0.95)        # on calibrated probabilities (models/calibration_d20.json)
MAX_UNSURE = 100             # daily budget: the most uncertain minutes first
EVERY = 10
# desks whose boxes overlap in the picture: someone at one can sit inside the other's box
NEIGHBOURS = (("1", "4"), ("1", "5"))
PAIR_EVERY = 3


def unsure(rows, temperature=1.0):
    """[(distance from 50/50, t)] for minutes where some desk is unsure once calibrated, most uncertain first."""
    out = []
    for r in rows:
        ps = [calibrated(p, temperature) for _, p in r["desks"].values() if p is not None]
        gaps = [abs(p - 0.5) for p in ps if UNSURE[0] < p < UNSURE[1]]
        if gaps:
            out.append((min(gaps), r["t"]))
    return sorted(out)


def spread(ts, n):
    """At most n of ts, evenly spaced, so a cut does not keep only the morning."""
    return list(ts) if len(ts) <= n else [ts[i * len(ts) // n] for i in range(n)]


def select(rows, reported, every=EVERY, temperature=1.0, max_unsure=MAX_UNSURE, extra=(), budget=None):
    """The minutes to label. With a budget, filled in priority order: reported mistakes (and two minutes
    either side) and `extra` (the control) always; then the most unsure; then the weak spots (desk 5,
    overlapping neighbours); then every `every`-th remaining minute."""
    must = set(extra)
    for t in reported:                       # the reported minute and two either side
        for d in (-120, -60, 0, 60, 120):
            must.add(min((r["t"] for r in rows), key=lambda x: abs(x - (t + d))))
    tiers = [[t for _, t in unsure(rows, temperature)[:max_unsure]]]
    weak = [r["t"] for r in rows if any(n == "5" and s != "vacant" for n, (s, _) in r["desks"].items())]
    pairs = [r["t"] for r in rows if r["t"] not in {*weak, *tiers[0]} and any(
        r["desks"].get(a, ["vacant"])[0] != "vacant" and r["desks"].get(b, ["vacant"])[0] != "vacant" for a, b in NEIGHBOURS)]
    tiers.append(sorted(weak + pairs[::PAIR_EVERY]))
    picked = set(must)
    for tier in tiers:
        new = [t for t in tier if t not in picked]
        picked.update(new if budget is None else spread(new, max(0, budget - len(picked))))
    rest = [r["t"] for r in rows if r["t"] not in picked]
    picked.update(rest[::every] if budget is None else spread(rest[::every], max(0, budget - len(picked))))
    return sorted(picked)


def latest(rows):
    """One answer per minute, the last one logged (an offline fill replaces a stale live answer), in time order."""
    return sorted({r["t"]: r for r in rows}.values(), key=lambda r: r["t"])


def logged(day):
    """The answers for the minutes that have a frame."""
    rows = [json.loads(l) for l in (day / "live_d20.jsonl").read_text().splitlines() if l.strip()]
    have = {int(f.stem.split("_")[1]) for f in (day / "frames").glob("d20_*.jpg")}
    return latest(r for r in rows if r["t"] in have)


def draw_boxes(frame, boxes):
    """The desk boxes (marked on the 2560-wide main stream) and their numbers on a 1280x720 frame, for labelling."""
    import cv2
    out = frame.copy()
    for n, b in boxes.items():
        x1, y1, x2, y2 = (round(v * 1280 / 2560) for v in b)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(str(n), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(out, (x1 + 4, y1 + 4), (x1 + 12 + tw, y1 + 12 + th), (0, 0, 0), -1)
        cv2.putText(out, str(n), (x1 + 8, y1 + 8 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def build(day, reported, extra=(), temperature=1.0, budget=None):
    """Write the review set for `day`: the picked minutes plus `extra` ones (see select). Returns them."""
    import cv2
    sys.path.insert(0, str(WS)); sys.path.insert(0, str(WS / "tools"))
    from presence import load_boxes
    rows = logged(day)
    ts = select(rows, reported, temperature=temperature, extra=extra, budget=budget)
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
    build(day, [int(h) * 3600 + int(m) * 60 for h, m in (a.split(":") for a in sys.argv[2:])],
          temperature=load_temperatures(WS / "models" / "calibration_d20.json")["seat"])


if __name__ == "__main__":
    main()
