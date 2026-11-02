# SeatSense — context for contributors

What the README does not say: why the system is shaped the way it is, the rules of the deployment it
was built for, and what must not break. Read this before changing models, rules or the schedule.

## The problem

An HR team wanted, per desk and per day: time at desk, time working, time present but not working
(phone, talking, idle). One existing CCTV camera covers six desks from above. Staff were informed,
HR owns the data, and reports must show desk numbers only — never names or faces.

The camera angle is the defining constraint: desks and partitions hide legs, monitors hide hands
and phones, and neighbouring desk boxes overlap in the picture. Most design decisions follow from it.

## How it works, in one paragraph

`seatsense.app` decodes the RTSP stream at 1280×720, and once a second scores each desk crop with
the **seat** classifier (is someone sitting here?) and, for occupied desks, the **activity**
classifier (working or not?). Hysteresis rules turn per-second scores into stable desk states; a
state per desk per minute is logged and drives the reports. A frame logger saves one full-resolution
frame per minute; every evening a review builder picks the frames worth labelling; every Sunday a
gated retrain replaces a model only if it is at least as good on a day neither model has seen.

## Decisions and their evidence

| Decision | Why (measured) |
|---|---|
| A **classifier per desk crop**, not "person box inside desk box" | Seated and standing people overlap a desk box by the same 0.77–0.85; no threshold separates them. The classifier cut missed sitters from 112 to 27 on a full labelled day. |
| **Pose / leg checks rejected** | Knees are almost never visible from this camera (a clearly standing person scored 0.19). |
| **Zero-shot "cell phone" detection rejected** | Equally confident on an AC remote, phones lying on desks, boxes. |
| **Full-resolution crops rejected** | 77.7% vs 81.2% at 720p on unseen days; no more phones caught. |
| **Seat: 0.75 to take a desk, 0.5 to keep it** | Leave-one-day-out, every fold independently chose 0.75; false "sitting" 121 → 79, total errors −14%. The lower bar to *keep* a desk stops a leaning sitter from dropping out. |
| **Activity: "not working" from 0.7** | Wrong flags −24% on unseen days; on the daily check samples it removed false flags without losing a single real catch. A wrong "not working" costs more (it accuses someone) than a missed one. |
| **Holds: 15 s to occupy, 10 s to vacate, 20 s to "not working"** | One wrong frame must never change a report; a glance at a phone must never count. The 10 s was the user's choice (faster departures). |
| **Missing data is "no data", never "vacant"** | Nobody may be shown absent while the system was blind. |

## How we evaluate (do not weaken this)

- **Folds by time, not by frame.** Neighbouring frames are near-identical; random splits leak.
  Within-day folds are half-hour blocks; the honest test for "next Monday" holds out whole days
  (`--cv --by-day`). The within-day number was optimistic for activity (86.7% vs ~82% on unseen days).
- **Final-epoch numbers only.** A best-epoch-on-test number (96.2%) was once reported; it was wrong.
- **The daily score uses only the random control sample**, picked without looking at the model, from
  a day the model has not trained on. Unsure and reported frames are for training, never for scoring.
- **Report precision/recall per class and per desk, and the minutes-at-desk error.** Accuracy hides
  weak desks (desk 5 once read 95.7% accurate while finding only 42% of its sitters).
- **A "fixed" reported mistake only counts after a retrain**, checked by `training/check_reported.py`.
  Reported frames are in the training set, so the unseen-day numbers remain the real proof.
- **Run-to-run GPU noise is about ±0.3 points.** Smaller differences are "no change".

## The learning loop

1. **Frame logger** — one frame per minute (09:30–19:00), with the live answer and its age; an
   answer older than 10 s (frozen or restarting detector) is treated as missing and re-scored offline.
2. **Review builder (19:10)** — a fixed budget (~30 min of labelling) of calibrated-unsure frames,
   known weak spots, user-reported mistakes (the "Report a mistake" button) and a random control.
3. **Labelling** — `labelling/` pages, keyboard-driven, pre-filled with the model's answer.
4. **Sunday gate** (`scripts/weekly.sh` + `training/gate.py`) — candidates trained without the newest
   labelled day; promoted only if at least as good on its control sample; old models kept in
   `models/history/`, `scripts/rollback.sh` restores them.

Uncertainty sampling never finds *confident* mistakes (e.g. a person standing bent over a desk,
scored 99.7% seated). Those only enter the data through users reporting them — keep that path easy.

## Labelling rules

- **Sitting** = seated in that desk's chair, including turned away, leaning back or on the phone.
- **Not sitting** = standing (even bent over the desk), walking past, leaning in from a neighbour,
  a visitor (hi-vis vest).
- Someone between two desks belongs to the desk they are working at.
- **Activity keys:** W working · P phone · T talking / turned away · I idle · U unclear (not trained on).

## Deployment rules

- **Privacy:** no frames, crops, labels, states, logs, accounts, certificates, `.env` or trained
  weights in git — ever. No IP addresses, site names or people's names in code or docs. Runtime data
  stays on the deployment machine; unpicked frames are deleted after 14 days.
  `scripts/privacy_check.py` enforces this (CI on every push; the pre-commit hook locally, with the
  `.env` values and `~/.config/seatsense/private-words.txt`). Run it with `--history` before making a
  fork public. A new screenshot needs a person to check it for faces and names, then its sha256 added to the
  script's list (never replace an old hash: the old picture stays in the history).
- **Shared GPU:** the deployment GPU also runs other projects. Start only with ≥ 3 GB free; train only
  when the GPU is free or on Sundays; never stop another project's processes.
- **Office hours:** 09:30–19:00, Monday–Saturday; nothing runs on Sundays except the gated retrain.
- **Restarts sign everyone out** (sessions live in memory) — restart only when needed, ideally
  outside office hours, and say so.
- **Never overwrite a running shell script in place** (bash reads scripts incrementally); write a new
  file and rename it.

## Known limits and open items

- A phone held behind a monitor is invisible to this camera; no model can learn it.
- The activity model is the weak part (not-working recall around two thirds) — labelled phone and
  talking examples are what moves it.
- The camera password appears in the decoder's command line on the host (`ps`); a small local RTSP
  relay would hide it.
- The self-made certificate makes each browser warn once; a company certificate would remove it.
- The person detector (Ultralytics YOLO) is AGPL-3.0 and only used for the overlay and the visitor
  check; a permissive detector would allow a closed-source deployment.
