# Generalisation to a new day (leave-one-day-out)

Half-hour folds keep the same days, people and light on both sides of the split. Here a whole day
is held out: the model trains on the other days and is tested on a day it has never seen, which is
the honest test for "next Monday". Run 23 Oct 2026 with `--cv --by-day`. The labelled days are a
full working day (12 Oct) and two hard-case review sets (17 and 22 Oct), so per-day numbers differ
in difficulty.

## Seat occupancy

| Held-out day | Trained on 12+17 only | Trained with 22 Oct added |
|---|---|---|
| 12 Oct (full day) | 95.9% | **96.7%** (sitters caught 96.3% → 98.1%) |
| 17 Oct (hard frames) | 91.3% | **94.4%** (false "sitting" 55 → 30) |
| 22 Oct (review set) | — | 98.6% |
| **All held-out desk checks** | 94.9% (box rule 94.6%) | **96.8%** (box rule 95.5%) |

Adding one labelled day improved both earlier days the model never saw. The in-day estimate was
97.8%, so the new-day cost for the seat model is about one point.

## Activity (working / not working)

| On days 12 + 17 (1,443 crops) | Trained without 22 Oct | Trained with 22 Oct |
|---|---|---|
| Accuracy | 81.3% | **82.1%** |
| Working recognised | 89.5% | 86.1% |
| Not working caught | 59.5% | **71.4%** |
| Precision of a "not working" flag | 68.1% | 65.9% |
| Balanced accuracy | 74.5% | **78.7%** |

**The gap matters:** the half-hour CV said 86.7%, but on an unseen day the activity model is at
about 81–82%, and on the hardest held-out set (22 Oct's review crops, trained on 12 + 17 only) 70.7%.
The in-day number was optimistic, exactly the leakage that blocked splits by day are meant to
expose. More days still help: one added day lifted not-working recall by 12 points, at a cost of
about 2 points of precision.

## Decision

Both models were retrained on 12 + 17 + 22 Oct and switched live on 23 Oct, because the held-out-day
results improved for both. The previous models are kept for rollback. Calibration temperatures were
refitted on these held-out predictions (seat 2.41, activity 2.93).
