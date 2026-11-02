#!/usr/bin/env python3
"""The daily loop: pick what to label from one logged day, then score the live models on it.

    python3 -m training.daily review recordings/20261022-log    # after 19:00 (timer: review today): build both review sets
    python3 -m training.daily score  recordings/20261022-log    # after labelling: that day's accuracy
    python3 -m training.daily backlog recordings/20261024-log   # a day that fell behind: label only its control

review: the seat review set (training/review_set.py: unsure minutes, desk 5, neighbours) plus
SEAT_CONTROL_N evenly spaced logged minutes as a plain control, capped at SEAT_BUDGET frames (minutes logged while seatsense/app.py was down get an
offline answer from the same seat model first) -> label on :8098 (labelling/label.py <day>). For every
desk the live system called occupied, the activity model scores the desk crop; the unsure ones
(closest to 50/50, at most MAX_UNSURE) and a random CONTROL_N of all of them are appended to
recordings/activity -> label on :8099 (labelling/label_activity.py recordings/activity).

Budget (27 Oct): about 30 min of labelling a day -> 90 seat frames + 60 activity crops. Uncapped, a full
day was ~290 frames + ~118 crops and the labelling stopped after two days.

backlog: for a day not labelled in time, keep only its control, reported moments and anything already
labelled (other label frames go to label_frames_skipped/, a manifest backup is kept), so the honest
score for that day is still possible without labelling the whole set.

score: accuracy on the control samples only -- picked without looking at the model, from a day
the models never trained on, so it is the honest number. Appended to models/daily_scores.jsonl.
Retraining (training/train_seat.py, training/train_activity.py) stays a separate step: the GPU is shared.
"""
import json
import random
import shutil
import sys
from pathlib import Path

from seatsense.paths import ROOT as WS
from training.evaluate import calibrated, load_temperatures
from training.review_set import spread
from seatsense.activity import NOT_AT   # (the live "not working" threshold; torch is imported only by the model)
ACT = WS / "recordings" / "activity"
SCORES = WS / "models" / "daily_scores.jsonl"
SEAT_BUDGET, SEAT_CONTROL_N = 90, 30
UNSURE = (0.15, 0.85)        # on calibrated probabilities, like the seat review set
MAX_UNSURE, CONTROL_N = 30, 30


def seat_control(rows, n=SEAT_CONTROL_N):
    """n logged minutes spread over the day, chosen without looking at the model."""
    return spread([r["t"] for r in rows], n)


def trim_manifest(manifest, day, labels):
    """Drop `day`'s activity crops that are neither control, reported nor already labelled."""
    return [m for m in manifest if m["dir"] != day or m.get("control") or m.get("reported") or m["file"] in labels]


STALE_S = 10   # a live answer older than this was not looking at the frame (seatsense/app.py skips its own log the same way)


def fresh(r):
    """Was the live answer current? Rows logged before ages were recorded (22 Oct and earlier) are trusted."""
    if "age" not in r:
        return True
    return r["age"] is not None and r["age"] <= STALE_S and r.get("online", True)


def missing(frames, rows):
    """Frame times with no current live answer (system down, frozen, offline or starting), in time order."""
    have = {r["t"] for r in rows if fresh(r)}
    return sorted(t for t in frames if t not in have)


def fill_missing(day):
    """Give frames logged while seatsense/app.py was down or stale an answer from the same seat model, so they
    can be reviewed and scored fairly. Raw per-frame answer (no 15 s hold, no visitor check); marked
    "offline" and appended, so it replaces the stale row (review_set.latest keeps the last)."""
    import cv2
    from seatsense.presence import load_boxes
    from seatsense.seat import SeatModel
    log = day / "live_d20.jsonl"
    rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()] if log.exists() else []
    ts = missing([int(f.stem.split("_")[1]) for f in (day / "frames").glob("d20_*.jpg")], rows)
    if not ts:
        return
    model = SeatModel("d20")
    boxes = {n: tuple(v / 2 for v in b) for n, b in load_boxes(WS / "config" / "zones" / "d20_focus.json").items()}
    with log.open("a") as fh:
        for t in ts:
            img = cv2.resize(cv2.imread(str(day / "frames" / f"d20_{t:05d}.jpg")), (1280, 720), interpolation=cv2.INTER_AREA)
            p = model.predict(img, boxes)
            fh.write(json.dumps({"t": t, "desks": {str(n): ["occupied" if p[n] >= 0.5 else "vacant", round(p[n], 3)] for n in p},
                                 "offline": True}) + "\n")
    print(f"filled {len(ts)} minutes without a live answer")


def activity_unsure(rows, temperature=1.0):
    """The seated desk-minutes the activity model is unsure about once calibrated, most uncertain first."""
    cal = [(abs(calibrated(r["p_not"], temperature) - 0.5), r) for r in rows]
    return [r for g, r in sorted(cal, key=lambda x: x[0]) if g < 0.5 - UNSURE[0]]


def reported_times(text):
    """reported.txt: one "HH:MM[:SS] [what was wrong]" per line, as the user saw it -> seconds of the day."""
    out = []
    for line in text.splitlines():
        head = line.split()[0] if line.strip() and not line.startswith("#") else ""
        if head[:2].isdigit():
            h, m, *sec = head.split(":")
            out.append(int(h) * 3600 + int(m) * 60 + (int(sec[0]) if sec else 0))
    return out


def reported_activity(reports, frame_ts, have, near_s=120):
    """(frame t, desk) for each working / not-working mistake the user reported, at the nearest
    logged frame, unless that crop is already in the set."""
    from seatsense.mistakes import CLAIM
    out = []
    for r in reports:
        if CLAIM.get(r["kind"], (None,))[0] != "activity" or not frame_ts:
            continue
        tf = min(frame_ts, key=lambda t: abs(t - r["t"]))
        if abs(tf - r["t"]) <= near_s and (tf, r["desk"]) not in have and (tf, r["desk"]) not in out:
            out.append((tf, r["desk"]))
    return out


def pick_activity(rows, seed, max_unsure=MAX_UNSURE, control_n=CONTROL_N, temperature=1.0):
    """rows: [{"t", "box", "p_not"}] for every seated desk-minute -> the picked ones, flagged."""
    unsure = activity_unsure(rows, temperature)[:max_unsure]
    control = random.Random(seed).sample(rows, min(control_n, len(rows)))
    keys = lambda rs: {(r["t"], r["box"]) for r in rs}
    u, c = keys(unsure), keys(control)
    return [{**r, "unsure": (r["t"], r["box"]) in u, "control": (r["t"], r["box"]) in c}
            for r in rows if (r["t"], r["box"]) in u | c]


def seat_score(pre, labels, control, desks=6):
    """Desk-minutes where the live answer matched the label, on labelled control minutes."""
    right = total = 0
    for t in map(str, control):
        if t in labels:
            right += sum((n in pre.get(t, [])) == (n in labels[t]) for n in range(1, desks + 1))
            total += desks
    return right, total


def activity_score(manifest, labels):
    out = {"right": 0, "total": 0, "working": [0, 0], "not": [0, 0]}
    for m in manifest:
        a = labels.get(m["file"])
        if not m.get("control") or a not in ("W", "P", "T", "I"):
            continue
        truth = "working" if a == "W" else "not"
        ok = (m["p_not"] >= NOT_AT) == (truth == "not")   # scored at the live system's threshold
        out["right"] += ok; out["total"] += 1
        out[truth][0] += ok; out[truth][1] += 1
    return out


def review(day):
    import cv2
    from seatsense.activity import ActivityModel
    from training.activity_crops import cut
    from seatsense.presence import load_boxes
    from training.review_set import build, logged, unsure

    temps = load_temperatures(WS / "models" / "calibration_d20.json")
    fill_missing(day)
    rows = logged(day)
    print(f"seat: {sum(bool(r.get('offline')) for r in rows)} of {len(rows)} minutes use an offline answer (live system down or stale)")
    control = seat_control(rows)
    rep = day / "reported.txt"   # mistakes the user saw live: confident errors uncertainty sampling never picks
    reported = reported_times(rep.read_text()) if rep.exists() else []
    build(day, reported, extra=control, temperature=temps["seat"], budget=SEAT_BUDGET)
    print(f"seat: {len(reported)} reported mistake(s) added with 2 minutes either side")
    print(f"seat: {len(unsure(rows, temps['seat']))} of {len(rows)} minutes genuinely unsure (calibrated, T={temps['seat']})")
    (day / "control_d20.json").write_text(json.dumps(control))

    boxes = load_boxes(day / "d20_focus.json")
    model = ActivityModel("d20")
    seated = []
    for r in rows:
        occ = [int(n) for n, (s, _) in r["desks"].items() if s != "vacant"]
        if not occ:
            continue
        img = cv2.resize(cv2.imread(str(day / "frames" / f"d20_{r['t']:05d}.jpg")), (1280, 720), interpolation=cv2.INTER_AREA)
        p = model.predict(img, {n: tuple(v / 2 for v in boxes[n]) for n in occ})
        seated += [{"t": r["t"], "box": n, "p_not": round(p[n], 4)} for n in occ]
    picked = pick_activity(seated, seed=day.name, temperature=temps["activity"])
    print(f"activity: {len(activity_unsure(seated, temps['activity']))} of {len(seated)} seated desk-minutes "
          f"genuinely unsure (calibrated, T={temps['activity']})")

    # working / not-working mistakes the user reported live: their crops go in too, flagged
    from seatsense.mistakes import parse
    reports = parse(rep.read_text()) if rep.exists() else []
    reported_keys = set(reported_activity(reports, [r["t"] for r in rows], set()))
    for e in picked:   # a reported crop already picked as unsure or control still gets its banner
        if (e["t"], e["box"]) in reported_keys:
            e["reported"] = True
    extra = reported_activity(reports, [r["t"] for r in rows], {(e["t"], e["box"]) for e in picked})
    for t, n in extra:
        img = cv2.resize(cv2.imread(str(day / "frames" / f"d20_{t:05d}.jpg")), (1280, 720), interpolation=cv2.INTER_AREA)
        p = model.predict(img, {n: tuple(v / 2 for v in boxes[n])})
        picked.append({"t": t, "box": n, "p_not": round(p[n], 4), "unsure": False, "control": False, "reported": True})
    print(f"activity: {len(extra)} reported working / not-working mistake(s) added")

    manifest = json.loads((ACT / "manifest.json").read_text())
    have = {m["file"] for m in manifest}
    added = 0
    for e in picked:
        name = f"{day.name}_{e['t']:05d}_{e['box']}.jpg"
        if name in have:
            continue
        img = cv2.imread(str(day / "frames" / f"d20_{e['t']:05d}.jpg"))
        cv2.imwrite(str(ACT / "crops" / name), cut(img, boxes[e["box"]]), [cv2.IMWRITE_JPEG_QUALITY, 90])
        manifest.append({"file": name, "dir": day.name, **e})
        added += 1
    (ACT / "manifest.json").write_text(json.dumps(manifest))
    print(f"activity: {len(seated)} seated desk-minutes -> {added} crops to label "
          f"({sum(e['unsure'] for e in picked)} unsure, {sum(e['control'] for e in picked)} control)")


def backlog(day):
    from training.review_set import logged, select
    rep, lab = day / "reported.txt", day / "labels_d20.json"
    keep = set(select(logged(day), reported_times(rep.read_text()) if rep.exists() else [],
                      extra=json.loads((day / "control_d20.json").read_text()), budget=0))
    keep |= {int(t) for t in (json.loads(lab.read_text()) if lab.exists() else {})}
    skipped = day / "label_frames_skipped"
    skipped.mkdir(exist_ok=True)
    moved = 0
    for f in sorted((day / "label_frames").glob("d20_*.jpg")):
        if int(f.stem.split("_")[1]) not in keep:
            f.rename(skipped / f.name)
            moved += 1
    man = json.loads((ACT / "manifest.json").read_text())
    shutil.copy(ACT / "manifest.json", ACT / f"manifest.before_backlog_{day.name}.json")
    kept = trim_manifest(man, day.name, json.loads((ACT / "labels.json").read_text()))
    (ACT / "manifest.json").write_text(json.dumps(kept))
    print(f"seat: {len(list((day / 'label_frames').glob('d20_*.jpg')))} frames left to label, {moved} moved to {skipped.name}/")
    print(f"activity: {sum(m['dir'] == day.name for m in kept)} crops of {day.name} left, {len(man) - len(kept)} dropped")


def score(day):
    pre = json.loads((day / "prelabels_d20.json").read_text())
    labels_f = day / "labels_d20.json"
    labels = json.loads(labels_f.read_text()) if labels_f.exists() else {}
    seat = seat_score(pre, labels, json.loads((day / "control_d20.json").read_text()))
    man = [m for m in json.loads((ACT / "manifest.json").read_text()) if m["dir"] == day.name]
    act = activity_score(man, json.loads((ACT / "labels.json").read_text()))
    entry = {"day": day.name, "seat": list(seat), "activity": act}
    old = [json.loads(l) for l in SCORES.read_text().splitlines()] if SCORES.exists() else []
    hist = [e for e in old if e["day"] != day.name] + [entry]
    SCORES.write_text("".join(json.dumps(e) + "\n" for e in sorted(hist, key=lambda e: e["day"])))
    pct = lambda a, b: f"{a / b:6.1%} ({b})" if b else "   — (0)"
    print(f"{'day':<14} {'seat (desk-min)':<18} {'activity':<16} {'working ok':<16} not-working caught")
    for e in sorted(hist, key=lambda e: e["day"]):
        a = e["activity"]
        print(f"{e['day']:<14} {pct(*e['seat']):<18} {pct(a['right'], a['total']):<16} {pct(*a['working']):<16} {pct(*a['not'])}")


if __name__ == "__main__":
    from datetime import date
    arg = sys.argv[2]
    day = WS / "recordings" / f"{date.today():%Y%m%d}-log" if arg == "today" else Path(arg)
    {"review": review, "score": score, "backlog": backlog}[sys.argv[1]](day)
