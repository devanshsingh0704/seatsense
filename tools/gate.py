#!/usr/bin/env python3
"""Should the retrained models replace the live ones? (tools/weekly.sh runs this.)

    python3 tools/gate.py recordings/20261026-log     # prints the models to promote, e.g. "seat activity"

`day` is the newest labelled day. The live models have never trained on it, and neither have the
candidates (models/*_d20_cand.*, trained on every other day), so both are scored on its control
sample: picked without looking at any model, scored as tools/daily.py scores a day (seat per frame at
0.5, activity at the live threshold). A candidate at least as good as the live model is promoted;
the decision is appended to models/retrain_log.jsonl.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS)); sys.path.insert(0, str(WS / "tools"))
from daily import ACT, activity_score, seat_score   # noqa: E402


def decide(live, cand):
    """(right, total) of each on the same samples -> promote the candidate?"""
    return cand[1] > 0 and cand[0] >= live[0]


def seat_answers(p_by_t):
    """{t: {desk: p_sitting}} -> {"t": [seated desks]}, the shape seat_score compares with labels."""
    return {str(t): sorted(n for n, p in ps.items() if p >= 0.5) for t, ps in p_by_t.items()}


def main():
    import cv2
    from activity import ActivityModel
    from presence import load_boxes
    from seat import SeatModel
    day = Path(sys.argv[1])
    control = json.loads((day / "control_d20.json").read_text())
    labels = json.loads((day / "labels_d20.json").read_text())
    act_labels = json.loads((ACT / "labels.json").read_text())
    crops = [m for m in json.loads((ACT / "manifest.json").read_text()) if m["dir"] == day.name and m.get("control")]
    boxes = {n: tuple(v / 2 for v in b) for n, b in load_boxes(day / "d20_focus.json").items()}
    frame = lambda t: cv2.resize(cv2.imread(str(day / "frames" / f"d20_{t:05d}.jpg")), (1280, 720), interpolation=cv2.INTER_AREA)

    scores = {}
    for name in ("d20", "d20_cand"):
        seat, act = SeatModel(name), ActivityModel(name)
        p = {t: seat.predict(frame(t), boxes) for t in control if str(t) in labels}
        a = activity_score([{**m, "p_not": act.predict(frame(m["t"]), {m["box"]: boxes[m["box"]]})[m["box"]]} for m in crops], act_labels)
        scores[name] = {"seat": seat_score(seat_answers(p), labels, control), "activity": (a["right"], a["total"])}
    promote = [m for m in ("seat", "activity") if decide(scores["d20"][m], scores["d20_cand"][m])]
    for m in ("seat", "activity"):
        print(f"{m}: live {scores['d20'][m]} candidate {scores['d20_cand'][m]} -> "
              f"{'promote' if m in promote else 'keep live'}", file=sys.stderr)
    with (WS / "models" / "retrain_log.jsonl").open("a") as fh:
        fh.write(json.dumps({"at": datetime.now().isoformat(timespec="seconds"), "day": day.name,
                             "scores": scores, "promote": promote}) + "\n")
    print(" ".join(promote))


if __name__ == "__main__":
    main()
