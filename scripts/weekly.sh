#!/usr/bin/env bash
# Sunday retrain, gated (desk-weekly-retrain.timer). The newest labelled day is held out: candidates are
# trained on every other day and scored against the live models on its control sample (training/gate.py).
# A candidate at least as good goes live as it is -- exactly the model that was tested -- after the live
# models are backed up to models/history/ (scripts/rollback.sh puts them back). The held-out day joins
# the training data next week. seatsense/app.py loads the new models at Monday's 09:30 start.
#   scripts/weekly.sh
set -eu
cd "$(dirname "$0")/.."
PY=${PY:-python3}
NEED_MB=4000   # training both models; the GPU is shared with other projects
LOG=models/retrain_log.jsonl

newest=$(for d in recordings/*-log; do if [ -f "$d/labels_d20.json" ] && [ -f "$d/control_d20.json" ]; then echo "$d"; fi; done | tail -1)
[ -n "$newest" ] || { echo "no labelled day with a control sample: nothing to do"; exit 0; }
# already gated, and nothing labelled since (a day kept for too few labelled samples is tried again once labelled)
if [ -f "$LOG" ] && grep -q "\"day\": \"$(basename "$newest")\"" "$LOG" \
   && [ "$LOG" -nt "$newest/labels_d20.json" ] && [ "$LOG" -nt recordings/activity/labels.json ]; then
    echo "$newest already gated and nothing labelled since: nothing to do"; exit 0
fi
if $PY -c 'import json, sys; sys.exit(sys.argv[1] not in json.load(open("models/seat_d20.json"))["trained_on"])' "$newest"; then
    echo "$newest is already trained on: no unseen day to test a candidate on"; exit 0
fi

until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge "$NEED_MB" ]; do
    [ "$(date +%u)" = 1 ] && { echo "GPU busy until Monday: skipped this week"; exit 0; }
    sleep 300
done

days=$(for d in recordings/*-day recordings/*-log; do if [ -f "$d/labels_d20.json" ] && [ "$d" != "$newest" ]; then echo "$d"; fi; done)
echo "candidates trained without $newest"
$PY -m training.train_seat $days --as d20_cand
$PY -m training.train_activity recordings/activity --res 720 --without "$(basename "$newest")" --as d20_cand
promote=$($PY -m training.gate "$newest")
if [ -n "$promote" ]; then
    backup=models/history/$(date +%Y%m%d-%H%M)
    mkdir -p "$backup"
    cp models/seat_d20.pt models/seat_d20.json models/activity_d20.pt models/activity_d20.json "$backup/"
    for m in $promote; do
        mv "models/${m}_d20_cand.pt" "models/${m}_d20.pt"
        mv "models/${m}_d20_cand.json" "models/${m}_d20.json"
    done
    echo "live now: $promote from the tested candidate; previous models in $backup (scripts/rollback.sh restores them)"
    $PY -m training.check_reported recordings/*-log   # do the new models now get the reported mistakes right?
else
    echo "kept the live models (see $LOG)"
fi
rm -f models/seat_d20_cand.* models/activity_d20_cand.*
