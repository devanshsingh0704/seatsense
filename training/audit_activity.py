#!/usr/bin/env python3
"""Second look at the activity crops the model confidently disagrees with.

    python3 -m training.audit_activity build    # -> recordings/activity_audit/, label it blind on :8099:
    python3 -m labelling.label_activity recordings/activity_audit
    python3 -m training.audit_activity apply    # the second answers replace the first (backup kept)

Held-out-day predictions (models/activity_d20_cv_720_byday.json, from train_activity.py --cv --by-day)
showed most errors are confident: 113 of 181 missed "not working" scored below 0.1, 74 of 109 wrong
flags above 0.9 (27 Oct). No threshold fixes those; either the label is wrong or the crop does not
show it. The audit set starts with no answers and without the model's score, so the second look is
blind. Desks 4 and 6 (the weakest) come first. Run apply while the main labelling page is closed: it
saves its whole label file and would overwrite the merge.
"""
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from seatsense.paths import ROOT as WS
ACT, AUDIT = WS / "recordings" / "activity", WS / "recordings" / "activity_audit"
CONFIDENT = (0.1, 0.9)
WEAK = (4, 6)


def pick(rows, lo=CONFIDENT[0], hi=CONFIDENT[1]):
    """Held-out rows where the model was sure of the opposite of the label; weak desks first."""
    out = [r for r in rows if (r["y"] and r["p_not"] < lo) or (not r["y"] and r["p_not"] > hi)]
    return sorted(out, key=lambda r: (r["box"] not in WEAK, r["box"], r["dir"], r["t"]))


def merge(old, new):
    """The second answer wins for every crop looked at again -> (labels, counts)."""
    n = {"same": 0, "changed": 0, "now_unclear": 0}
    for f, a in new.items():
        if a == old.get(f):
            n["same"] += 1
        else:
            n["changed"] += 1
            n["now_unclear"] += a == "U"
    return {**old, **new}, n


def build(cv_rows):
    rows = pick(json.loads(cv_rows.read_text()))
    AUDIT.mkdir(exist_ok=True)
    if not (AUDIT / "crops").exists():
        (AUDIT / "crops").symlink_to("../activity/crops")   # relative: survives a moved workspace
    (AUDIT / "manifest.json").write_text(json.dumps([{k: r[k] for k in ("file", "dir", "t", "box")} for r in rows]))
    if not (AUDIT / "labels.json").exists():
        (AUDIT / "labels.json").write_text("{}")
    by_desk = {b: sum(r["box"] == b for r in rows) for b in sorted({r["box"] for r in rows})}
    print(f"{len(rows)} crops to look at again, by desk {by_desk}; label them on :8099 "
          f"(python3 -m labelling.label_activity {AUDIT.relative_to(WS)})")


def apply():
    new = json.loads((AUDIT / "labels.json").read_text())
    old = json.loads((ACT / "labels.json").read_text())
    backup = ACT / f"labels.before_audit_{datetime.now():%Y%m%d-%H%M}.json"
    shutil.copy(ACT / "labels.json", backup)
    merged, n = merge(old, new)
    (ACT / "labels.json").write_text(json.dumps(merged, indent=1))
    man = json.loads((AUDIT / "manifest.json").read_text())
    print(f"looked at again {len(new)} of {len(man)}: {n['same']} kept, {n['changed']} changed "
          f"({n['now_unclear']} now unclear, left out of training); backup {backup.name}")
    for b in sorted({m["box"] for m in man}):
        files = [m["file"] for m in man if m["box"] == b and m["file"] in new]
        print(f"  desk {b}: {sum(new[f] != old.get(f) for f in files)} of {len(files)} changed")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(Path(sys.argv[2]) if len(sys.argv) > 2 else WS / "models" / "activity_d20_cv_720_byday.json")
    else:
        apply()
