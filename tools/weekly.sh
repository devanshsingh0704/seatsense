#!/usr/bin/env bash
# Sunday retrain, gated (desk-weekly-retrain.timer). The newest labelled day is held out: candidates are
# trained on every other day and scored against the live models on its control sample (tools/gate.py).
# Only a model at least as good as the live one is retrained on everything (tools/retrain.sh, which
# keeps the old models in models/history/). live.py loads the new ones at Monday's 09:30 start.
#   tools/weekly.sh
set -eu
cd "$(dirname "$0")/.."
PY=${PY:-python3}
NEED_MB=4000   # training both models; the GPU is shared with other projects

newest=$(for d in recordings/*-log; do if [ -f "$d/labels_d20.json" ] && [ -f "$d/control_d20.json" ]; then echo "$d"; fi; done | tail -1)
[ -n "$newest" ] || { echo "no labelled day with a control sample: nothing to do"; exit 0; }
if $PY -c 'import json, sys; sys.exit(sys.argv[1] not in json.load(open("models/seat_d20.json"))["trained_on"])' "$newest"; then
    echo "$newest is already trained on: no new labelled day since the last retrain"; exit 0
fi

until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge "$NEED_MB" ]; do
    [ "$(date +%u)" = 1 ] && { echo "GPU busy until Monday: skipped this week"; exit 0; }
    sleep 300
done

days=$(for d in recordings/*-day recordings/*-log; do if [ -f "$d/labels_d20.json" ] && [ "$d" != "$newest" ]; then echo "$d"; fi; done)
echo "candidates trained without $newest"
$PY tools/train_seat.py $days --as d20_cand
$PY tools/train_activity.py recordings/activity --res 720 --without "$(basename "$newest")" --as d20_cand
promote=$($PY tools/gate.py "$newest")
rm -f models/seat_d20_cand.* models/activity_d20_cand.*
if [ -n "$promote" ]; then tools/retrain.sh $promote; else echo "kept the live models (see models/retrain_log.jsonl)"; fi
