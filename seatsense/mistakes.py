"""'Report a mistake' from the live page: one line per wrong desk in recordings/<day>-log/reported.txt,

    16:39:41 desk 3 not_seated_shown_occupied

The daily review (training/daily.py) adds each reported moment, two minutes either side, to the next
labelling set: the confident mistakes that uncertainty sampling never picks. After labelling and a
retrain (training/check_reported.py), status() tells whether the new model gets that moment right.
"""
import json

KINDS = {
    "seated_shown_vacant": "someone seated, shown vacant",
    "not_seated_shown_occupied": "nobody seated (standing / passing / leaning in), shown occupied",
    "working_shown_not": "working, shown not working",
    "not_working_shown_working": "not working (phone / talking / idle), shown working",
}
# what each report says the truth was: (question, answer) -- seat: someone seated? activity: not working?
CLAIM = {"seated_shown_vacant": ("seat", True), "not_seated_shown_occupied": ("seat", False),
         "not_working_shown_working": ("activity", True), "working_shown_not": ("activity", False)}


def lines(reports, desks, now):
    """reports: {desk: kind or "ok"} from the form -> one line per wrong desk, desk order."""
    out = []
    for desk, kind in reports.items():
        if kind == "ok":
            continue
        if not str(desk).isdigit() or int(desk) not in desks or kind not in KINDS:
            raise ValueError("unknown desk or mistake")
        out.append((int(desk), f"{now:%H:%M:%S} desk {int(desk)} {kind}"))
    if not out:
        raise ValueError("no desk marked wrong")
    return [line for _, line in sorted(out)]


def parse(text):
    """reported.txt -> [{"t", "desk", "kind", "what"}]; free-text lines (no known kind) keep their words."""
    out = []
    for raw in text.splitlines():
        p = raw.split()
        if len(p) < 3 or not p[0][:2].isdigit() or p[1] != "desk" or not p[2].isdigit():
            continue
        h, m, *sec = p[0].split(":")
        kind = p[3] if len(p) > 3 and p[3] in KINDS else None
        out.append({"t": int(h) * 3600 + int(m) * 60 + (int(sec[0]) if sec else 0), "desk": int(p[2]), "kind": kind,
                    "what": KINDS[kind] if kind else " ".join(p[3:])})
    return out


def status(report, labelled, check):
    """labelled: the true answer to the report's question from the labels (None = not labelled yet);
    check: the retrained model's answer for that moment (None = not retrained since)."""
    question, claim = CLAIM.get(report["kind"], (None, None))
    if labelled is None:
        s = "waiting for labelling"
    elif question and labelled != claim:
        s = "labelled: the system was right"
    elif check is None:
        s = "labelled, waiting for retrain"
    else:
        s = "fixed" if check == labelled else "not fixed yet"
    return {**report, "question": question, "status": s}


def load_json(path, default):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def collect(day, manifest, act_labels, near_s=120):
    """Every report of one day folder with its status. day: recordings/<date>-log; manifest and
    act_labels: recordings/activity (the working / not-working crops and their labels)."""
    rep = day / "reported.txt"
    if not rep.exists():
        return []
    log = day / "live_d20.jsonl"
    ts = [json.loads(l)["t"] for l in log.read_text().splitlines() if l.strip()] if log.exists() else []
    seat = load_json(day / "labels_d20.json", {})
    check = load_json(day / "reported_check.json", {})
    crops = {(m["t"], m["box"]): m["file"] for m in manifest if m["dir"] == day.name}
    out = []
    for r in parse(rep.read_text()):
        tf = min(ts, key=lambda t: abs(t - r["t"]), default=None)
        tf = tf if tf is not None and abs(tf - r["t"]) <= near_s else None
        question = CLAIM.get(r["kind"], (None,))[0]
        labelled = None
        if tf is not None and question == "seat" and str(tf) in seat:
            labelled = r["desk"] in seat[str(tf)]
        elif tf is not None and question == "activity":
            a = act_labels.get(crops.get((tf, r["desk"])))
            labelled = None if a not in ("W", "P", "T", "I") else a != "W"
        c = check.get(str(tf), {}).get(str(r["desk"]), {}).get(question) if question else None
        out.append({**status(r, labelled, c), "t_frame": tf})
    return out
