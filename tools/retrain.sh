#!/usr/bin/env bash
# Retrain both models on everything labelled so far. Runs on the GPU laptop, only when the GPU is
# free (it is shared) -- never from a timer. Keeps the current models in models/history/<stamp>/
# so a worse day can be rolled back; live.py picks the new ones up on its next restart.
#   tools/retrain.sh
set -eu
cd "$(dirname "$0")/.."
PY=${PY:-python3}
stamp=$(date +%Y%m%d-%H%M)
mkdir -p "models/history/$stamp"
cp models/seat_d20.pt models/seat_d20.json models/activity_d20.pt models/activity_d20.json "models/history/$stamp/"
days=$(for d in recordings/*-day recordings/*-log; do [ -f "$d/labels_d20.json" ] && echo "$d"; done)
echo "seat model on: $days"
$PY tools/train_seat.py $days
$PY tools/train_activity.py recordings/activity --res 720
echo "done; old models in models/history/$stamp"
