#!/usr/bin/env python3
"""After a retrain: does the new model now get each reported mistake right?

    python3 -m training.check_reported recordings/*-log      # scripts/retrain.sh runs this after training

For every line of a day's reported.txt, the nearest logged frame is scored by the current seat and
activity models (per frame, no 15 s / 20 s hold) and written to that day's reported_check.json,
which mistakes.collect() turns into "fixed" / "not fixed yet" on the Reports page.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

from seatsense.paths import ROOT as WS


def main():
    import cv2
    from seatsense.activity import ActivityModel
    from seatsense.mistakes import parse
    from seatsense.presence import load_boxes
    from seatsense.seat import SeatModel
    seat, act = SeatModel("d20"), ActivityModel("d20")
    for day in map(Path, sys.argv[1:]):
        rep, log = day / "reported.txt", day / "live_d20.jsonl"
        if not rep.exists() or not log.exists():
            continue
        frames = {int(f.stem.split("_")[1]) for f in (day / "frames").glob("d20_*.jpg")}
        ts = sorted({json.loads(l)["t"] for l in log.read_text().splitlines() if l.strip()} & frames)
        boxes = {n: tuple(v / 2 for v in b) for n, b in load_boxes(WS / "config" / "zones" / "d20_focus.json").items()}
        out = {"_checked_at": datetime.now().isoformat(timespec="seconds")}
        for r in parse(rep.read_text()):
            tf = min(ts, key=lambda t: abs(t - r["t"]), default=None)
            if tf is None or abs(tf - r["t"]) > 120 or r["desk"] not in boxes:
                continue
            img = cv2.resize(cv2.imread(str(day / "frames" / f"d20_{tf:05d}.jpg")), (1280, 720), interpolation=cv2.INTER_AREA)
            b = {r["desk"]: boxes[r["desk"]]}
            ps, pn = seat.predict(img, b)[r["desk"]], act.predict(img, b)[r["desk"]]
            out.setdefault(str(tf), {})[str(r["desk"])] = {"seat": ps >= 0.5, "activity": pn >= 0.5,
                                                           "p_sitting": round(ps, 3), "p_not": round(pn, 3)}
        (day / "reported_check.json").write_text(json.dumps(out, indent=1))
        print(f"{day.name}: checked {len(out) - 1} reported moment(s)")


if __name__ == "__main__":
    main()
