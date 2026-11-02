#!/usr/bin/env python3
"""Delete camera frames nobody will use: the data-retention rule (README, Privacy).

    python3 -m scripts.prune --dry-run     # what would go
    python3 -m scripts.prune               # nightly, after the 19:10 review (desk-daily-review.service)

The frame logger saves one frame a minute; about 90 a day are picked for labelling. In every dated
recordings folder older than KEEP_DAYS, a frame is deleted unless it is labelled, in that day's
labelling set (label_frames/) or behind an activity crop. Picked frames stay, labelled or not, so a
day that fell behind can still be labelled; the frames backlog trimming set aside go with the rest.
Desk states (states/), labels, crops and models are never touched.
"""
import json
import shutil
import sys
from datetime import date, datetime

from seatsense.paths import ROOT as WS
KEEP_DAYS = 14


def old_enough(name, today, days=KEEP_DAYS):
    try:
        return (today - datetime.strptime(name[:8], "%Y%m%d").date()).days >= days
    except ValueError:
        return False


def doomed(frames, keep):
    """frames: d20_<second>.jpg names -> those whose second is not in keep."""
    return [f for f in frames if int(f.split("_")[1][:5]) not in keep]


def main():
    dry = "--dry-run" in sys.argv
    rec = WS / "recordings"
    manifest = json.loads((rec / "activity" / "manifest.json").read_text())
    total = 0
    for day in sorted(p for p in rec.iterdir() if (p / "frames").is_dir() and old_enough(p.name, date.today())):
        lab = day / "labels_d20.json"
        keep = {int(t) for t in (json.loads(lab.read_text()) if lab.exists() else {})}
        keep |= {int(f.stem.split("_")[1]) for f in (day / "label_frames").glob("d20_*.jpg")}
        keep |= {m["t"] for m in manifest if m["dir"] == day.name}
        gone = doomed(sorted(f.name for f in (day / "frames").glob("d20_*.jpg")), keep)
        skipped = day / "label_frames_skipped"
        print(f"{day.name}: {len(gone)} frames {'would go' if dry else 'deleted'}, "
              f"{len(list((day / 'frames').glob('d20_*.jpg'))) - len(gone)} kept"
              + (f", {skipped.name}/ removed" if skipped.exists() else ""))
        if not dry:
            for f in gone:
                (day / "frames" / f).unlink()
            shutil.rmtree(skipped, ignore_errors=True)
        total += len(gone)
    print(f"{total} frames {'would go' if dry else 'deleted'} (older than {KEEP_DAYS} days, never picked for labelling)")


if __name__ == "__main__":
    main()
