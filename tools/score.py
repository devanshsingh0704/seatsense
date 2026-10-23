#!/usr/bin/env python3
"""Score the desk rules against hand labels from a test recording. CPU only, seconds.

    python3 tools/score.py recordings/20261011-153440 [--min-inside 0.85] [--enter 15] [--leave 20]

Replays the detections saved by extract.py through the real presence.py rules (not a
copy), so a rule change is re-scored without running the detector again. Labels are
labels_<cam>.json: {"<second>": [boxes where someone is really SITTING], ...}.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from presence import MIN_INSIDE, Desk, assign, load_boxes   # noqa: E402


def replay(frames, boxes, min_inside=MIN_INSIDE, enter_s=15, leave_s=20, gap_s=3):
    """frames: [{"t": s, "persons": [[x1, y1, x2, y2, conf], ...]}] -> {t: {box: state}}."""
    desks = {n: Desk(enter_s, leave_s, gap_s) for n in boxes}
    out = {}
    for f in frames:
        owners = assign([p[:4] for p in f["persons"]], boxes, min_inside)
        for n, d in desks.items():
            d.update(bool(owners[n]), f["t"])
        out[f["t"]] = {n: d.state for n, d in desks.items()}
    return out


def score(states, labels, boxes):
    r = {"checks": 0, "correct": 0, "false_occupied": 0, "false_vacant": 0,
         "by_box": {n: {"correct": 0, "false_occupied": 0, "false_vacant": 0} for n in boxes}}
    for t, sitting in labels.items():
        if t not in states:
            continue
        for n in boxes:
            said, truth = states[t][n] == "occupied", n in sitting
            kind = "correct" if said == truth else "false_occupied" if said else "false_vacant"
            r["checks"] += 1
            r[kind] += 1
            r["by_box"][n][kind] += 1
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording")
    ap.add_argument("--min-inside", type=float, default=MIN_INSIDE)
    ap.add_argument("--enter", type=float, default=15)
    ap.add_argument("--leave", type=float, default=20)
    a = ap.parse_args()
    rec = Path(a.recording)
    for det in sorted(rec.glob("detections_*.jsonl")):
        cam = det.stem.split("_", 1)[1]
        lab = rec / f"labels_{cam}.json"
        if not lab.exists():
            print(f"{cam}: no labels yet ({lab.name})")
            continue
        boxes = load_boxes(rec / f"{cam}_focus.json")   # the boxes in force when it was recorded
        frames = [json.loads(l) for l in det.read_text().splitlines()]
        labels = {int(t): v for t, v in json.loads(lab.read_text()).items()}
        r = score(replay(frames, boxes, a.min_inside, a.enter, a.leave), labels, boxes)
        pct = 100 * r["correct"] / r["checks"] if r["checks"] else 0
        print(f"{cam}: {r['correct']}/{r['checks']} box checks right ({pct:.1f}%), "
              f"false occupied {r['false_occupied']}, false vacant {r['false_vacant']}")
        for n, b in r["by_box"].items():
            print(f"   box {n}: right {b['correct']}, false occupied {b['false_occupied']}, false vacant {b['false_vacant']}")


if __name__ == "__main__":
    main()
