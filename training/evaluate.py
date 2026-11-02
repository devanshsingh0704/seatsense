#!/usr/bin/env python3
"""Deeper evaluation of the saved cross-validation predictions. CPU only, seconds, no training.

    python3 -m training.evaluate models/seat_d20_cv_rows.json models/activity_d20_cv_720.json > docs/evaluation.md

Reads every held-out answer the --cv runs saved (training/train_seat.py, training/train_activity.py)
and reports what plain accuracy hides:
  - precision / recall / F1 and the confusion matrix, overall and per desk (accuracy is inflated
    by the many easy vacant desk-frames);
  - the business number: minutes at desk per desk and day, labelled vs predicted (one frame per
    minute, so a frame is a minute);
  - calibration: is "90% sure" right 90% of the time? Temperature scaling, fitted on the other
    folds only, with ECE and a reliability table;
  - the precision / recall trade-off of the "not working" threshold.
Also writes calibration_d20.json (one temperature per model, fitted on all held-out answers) next
to the CV rows; the daily review builder reads it.
Runs made before 23 Oct saved only probabilities rounded to 4 decimals, so their logits are
recovered and clamped at +-9.2; newer runs save the raw logit, which is used when present.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from training.train_seat import fold_of

EPS = 1e-4
TEMPERATURES = [round(0.2 * 1.05 ** k, 4) for k in range(95)]   # 0.2 .. ~20, log-spaced


def binary_metrics(y, pred):
    tp = sum(a and b for a, b in zip(y, pred)); fn = sum(a and not b for a, b in zip(y, pred))
    fp = sum(b and not a for a, b in zip(y, pred)); tn = len(y) - tp - fn - fp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "precision": precision, "recall": recall, "f1": f1,
            "accuracy": (tp + tn) / len(y) if y else 0.0}


def minutes_error(rows, threshold=0.5):
    """rows: {"dir", "box", "y", "p"} at one frame per minute -> {(dir, box): minutes labelled vs predicted}."""
    out = defaultdict(lambda: {"labelled": 0, "predicted": 0, "error": 0, "missed": 0, "extra": 0})
    for r in rows:
        o, pred = out[(r["dir"], r["box"])], r["p"] >= threshold
        o["labelled"] += r["y"]; o["predicted"] += pred
        o["missed"] += r["y"] and not pred; o["extra"] += pred and not r["y"]
        o["error"] = o["predicted"] - o["labelled"]
    return dict(out)


def logit(p):
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def scale(z, t):
    return 1 / (1 + math.exp(-z / t))


def nll(z, y, t):
    return -sum(math.log(max(scale(a, t) if b else 1 - scale(a, t), 1e-12)) for a, b in zip(z, y)) / len(y)


def fit_temperature(z, y):
    return min(TEMPERATURES, key=lambda t: nll(z, y, t))


def ece(p, y, bins=10):
    """Expected calibration error and the reliability table [(bin, n, mean confidence, observed rate)]."""
    groups = defaultdict(list)
    for a, b in zip(p, y):
        groups[min(int(a * bins), bins - 1)].append((a, b))
    table, err = [], 0.0
    for k in sorted(groups):
        g = groups[k]
        conf, freq = sum(a for a, _ in g) / len(g), sum(b for _, b in g) / len(g)
        err += len(g) / len(p) * abs(conf - freq)
        table.append((k, len(g), conf, freq))
    return err, table


def calibrated(p, t):
    return None if p is None else scale(logit(p), t)


def load_temperatures(path):
    """The fitted temperatures (written next to the CV rows by this tool); none yet = no calibration."""
    try:
        t = json.loads(path.read_text())
        return {"seat": t["seat"], "activity": t["activity"]}
    except FileNotFoundError:
        return {"seat": 1.0, "activity": 1.0}


def pr_table(p, y, thresholds):
    return [{"threshold": t, **binary_metrics(y, [a >= t for a in p])} for t in thresholds]


def brier(p, y):
    return sum((a - b) ** 2 for a, b in zip(p, y)) / len(y)


def score(r):
    """The raw logit when the run saved one, else recovered from the rounded probability."""
    return r["z"] if r.get("z") is not None else logit(r["p"])


def cross_fit(rows):
    """Calibrated probabilities: each fold scaled by the temperature fitted on the other folds."""
    temps, out = {}, {}
    for f in sorted({fold_of(r["t"]) for r in rows}):
        rest = [r for r in rows if fold_of(r["t"]) != f]
        temps[f] = fit_temperature([score(r) for r in rest], [r["y"] for r in rest])
        for i, r in enumerate(rows):
            if fold_of(r["t"]) == f:
                out[i] = scale(score(r), temps[f])
    return [out[i] for i in range(len(rows))], temps


# ---------- report ----------

def pct(x):
    return f"{x:.1%}"


def metrics_table(groups, positive):
    lines = [f"| | n | {positive} precision | {positive} recall | F1 | accuracy | TP | FN | FP | TN |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, (y, pred) in groups:
        m = binary_metrics(y, pred)
        lines.append(f"| {name} | {len(y)} | {pct(m['precision'])} | {pct(m['recall'])} | {m['f1']:.3f} | {pct(m['accuracy'])} "
                     f"| {m['tp']} | {m['fn']} | {m['fp']} | {m['tn']} |")
    return "\n".join(lines)


def calibration_section(rows, label):
    y = [r["y"] for r in rows]
    raw = [r["p"] for r in rows]
    cal, temps = cross_fit(rows)
    e_raw, _ = ece(raw, y); e_cal, table = ece(cal, y)
    out = [f"**{label}.** Temperature per held-out fold (fitted on the other folds): "
           + ", ".join(f"{t:.2f}" for t in temps.values()) + " (above 1 = the raw model is over-confident).", "",
           "| | ECE | Brier | log-loss |", "|---|---|---|---|",
           f"| raw | {e_raw:.3f} | {brier(raw, y):.3f} | {nll([score(r) for r in rows], y, 1.0):.3f} |",
           f"| temperature-scaled | {e_cal:.3f} | {brier(cal, y):.3f} | {nll([logit(a) for a in cal], y, 1.0):.3f} |", "",
           "Reliability after scaling (the model's stated probability vs how often it was really true):", "",
           "| stated probability | n | mean stated | observed |", "|---|---|---|---|"]
    out += [f"| {k / 10:.1f}–{(k + 1) / 10:.1f} | {n} | {pct(c)} | {pct(f)} |" for k, n, c, f in table]
    return "\n".join(out)


def main():
    seat_rows = [{"dir": Path(r["dir"]).name, "box": r["box"], "t": r["t"], "y": r["sitting"], "p": r["p_sitting"], "z": r.get("logit"), "rule": r["rule"]}
                 for r in json.loads(Path(sys.argv[1]).read_text())]
    act_rows = [{"dir": r["dir"], "box": r["box"], "t": r["t"], "y": r["y"], "p": r["p_not"], "z": r.get("logit"), "why": r["why"]}
                for r in json.loads(Path(sys.argv[2]).read_text())]
    # the temperatures the review builder uses (training/daily.py); refreshed whenever this is re-run after a new --cv
    temps = {"seat": fit_temperature([score(r) for r in seat_rows], [r["y"] for r in seat_rows]),
             "activity": fit_temperature([score(r) for r in act_rows], [r["y"] for r in act_rows]),
             "fitted_on": [Path(a).name for a in sys.argv[1:3]]}
    (Path(sys.argv[1]).parent / "calibration_d20.json").write_text(json.dumps(temps, indent=1))
    days = sorted({r["dir"] for r in seat_rows})
    natural = days[0]                                  # the full labelled day; later days are hard-case review sets
    nat = [r for r in seat_rows if r["dir"] == natural]
    pred = lambda rs, key="p": [r[key] >= 0.5 if key == "p" else bool(r[key]) for r in rs]
    out = ["# Evaluation", "",
           "Generated by `training/evaluate.py` from the held-out predictions of 4-fold cross-validation "
           "(folds = half-hour blocks of the day). Every number is on frames the model did not train on.", "",
           "## Seat occupancy (positive class = someone seated)", "",
           f"`{natural}` is a full working day labelled minute by minute (the natural mix). The other days are "
           "**review sets of hard frames** picked because the live model was unsure there, so their numbers "
           "are deliberately pessimistic and not comparable.", ""]
    groups = []
    for d in days:
        rs = [r for r in seat_rows if r["dir"] == d]
        groups += [(f"{d} · seat classifier", ([r["y"] for r in rs], pred(rs))),
                   (f"{d} · box rule (baseline)", ([r["y"] for r in rs], pred(rs, "rule")))]
    out += [metrics_table(groups, "seated"), "", f"Per desk, `{natural}`, seat classifier:", ""]
    out += [metrics_table([(f"desk {b}", ([r["y"] for r in nat if r["box"] == b], pred([r for r in nat if r["box"] == b])))
                           for b in sorted({r["box"] for r in nat})], "seated")]

    out += ["", "## Minutes at desk: the number HR actually reads", "",
            f"`{natural}`, one frame per minute, so labelled frames = minutes really seated. "
            "Before the 15 s / 10 s hold (which only removes flicker between frames).", "",
            "| desk | labelled min | classifier min | error | missed | extra | box rule min | box rule error |",
            "|---|---|---|---|---|---|---|---|"]
    clf = minutes_error(nat)
    rule = minutes_error([{**r, "p": float(r["rule"])} for r in nat])
    for (d, b), o in sorted(clf.items(), key=lambda kv: kv[0][1]):
        ro = rule[(d, b)]
        out.append(f"| {b} | {o['labelled']} | {o['predicted']} | {o['error']:+d} | {o['missed']} | {o['extra']} | {ro['predicted']} | {ro['error']:+d} |")
    lab = sum(o["labelled"] for o in clf.values())
    abs_c = sum(abs(o["error"]) for o in clf.values()); abs_r = sum(abs(o["error"]) for o in rule.values())
    out += ["", f"Total over all desks: {lab} labelled seated minutes; absolute error **{abs_c} min ({abs_c / max(1, lab):.1%})** "
            f"for the classifier vs {abs_r} min ({abs_r / max(1, lab):.1%}) for the box rule. Missed and extra minutes "
            "partly cancel within a desk, so the per-desk error is smaller than the frame error count."]

    y = [r["y"] for r in act_rows]
    out += ["", "## Activity (positive class = not working) — beta", "",
            metrics_table([("all seated people", (y, pred(act_rows)))]
                          + [(f"desk {b}", ([r["y"] for r in act_rows if r["box"] == b], pred([r for r in act_rows if r["box"] == b])))
                             for b in sorted({r["box"] for r in act_rows})], "not working"), "",
            "Caught, by labelled reason:", "", "| reason | caught |", "|---|---|"]
    for why, name in (("P", "on the phone"), ("T", "talking / turned away"), ("I", "idle"), ("W", "working (wrongly flagged)")):
        g = [r for r in act_rows if r["why"] == why]
        out.append(f"| {name} | {sum(r['p'] >= 0.5 for r in g)}/{len(g)} |")
    out += ["", "Threshold trade-off for flagging \"not working\" (raw model probability; the live system uses 0.5):", "",
            "| threshold | precision | recall | F1 | flagged |", "|---|---|---|---|---|"]
    for r in pr_table([r["p"] for r in act_rows], y, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        out.append(f"| {r['threshold']:.1f} | {pct(r['precision'])} | {pct(r['recall'])} | {r['f1']:.3f} | {r['tp'] + r['fp']} |")
    out += ["", "Wrongly calling someone \"not working\" costs more than missing a phone glance, so a threshold "
            "should be chosen for precision first; scores here are agreement with one labeller.",
            "", "## Calibration", "", calibration_section(seat_rows, "Seat"), "", calibration_section(act_rows, "Activity"), ""]
    print("\n".join(out))


if __name__ == "__main__":
    main()
