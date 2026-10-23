"""Desk work reports from the per-minute state log (states/<YYYY-MM-DD>.jsonl).

Each log line is {"t": second_of_day, "desks": {"1": state, ...}} with state one of
"working", "not" (present, not working), "occupied" (at desk, before the 3-state model)
or "vacant". Reports count office-hour minutes only; a minute with no sample (detector
off) is "no data", never vacant.
"""
import datetime
import json

OPEN_S, CLOSE_S = 9 * 3600 + 1800, 19 * 3600     # office hours 09:30-19:00 (changed from 09:00-18:00 on 22 Oct)
MINUTES = (CLOSE_S - OPEN_S) // 60
AT_DESK = ("working", "not", "occupied")


def load(path):
    try:
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    except FileNotFoundError:
        return []


def day_report(samples, desks):
    slots = {d: ["nodata"] * MINUTES for d in desks}
    for s in samples:
        i = (s["t"] - OPEN_S) // 60
        if 0 <= i < MINUTES:
            for d in desks:
                if d in s["desks"]:
                    slots[d][i] = s["desks"][d]
    out = {}
    for d, row in slots.items():
        cnt = {k: row.count(k) for k in ("working", "not", "occupied", "vacant", "nodata")}
        left = sum(1 for a, b in zip(row, row[1:]) if a in AT_DESK and b == "vacant")
        longest = run = 0
        for st in row:
            run = run + 1 if st in AT_DESK else 0
            longest = max(longest, run)
        segs = []
        for i, st in enumerate(row):
            if segs and segs[-1][0] == st:
                segs[-1][2] = i + 1
            else:
                segs.append([st, i, i + 1])
        out[d] = {"at_desk": cnt["working"] + cnt["not"] + cnt["occupied"], "working": cnt["working"], "not": cnt["not"],
                  "vacant": cnt["vacant"], "no_data": cnt["nodata"], "times_left": left, "longest": longest, "segments": segs}
    return out


def week_dates(date):
    d = datetime.date.fromisoformat(date)
    monday = d - datetime.timedelta(days=d.weekday())
    return [(monday + datetime.timedelta(days=i)).isoformat() for i in range(5)]


def to_csv(rep, date):
    lines = ["date,desk,at_desk_min,working_min,not_working_min,vacant_min,no_data_min,times_left,longest_min"]
    for d, r in rep.items():
        lines.append(f"{date},{d},{r['at_desk']},{r['working']},{r['not']},{r['vacant']},{r['no_data']},{r['times_left']},{r['longest']}")
    return "\n".join(lines) + "\n"
