#!/usr/bin/env bash
# Retrain the models on everything labelled so far. Runs on the GPU laptop when the GPU is free (it is
# shared). By hand only -- the Sunday job (scripts/weekly.sh) puts the gate-tested candidate live instead. Keeps the
# current models in models/history/<stamp>/ so a worse day can be rolled back; seatsense/app.py picks the new
# ones up on its next restart.
#   scripts/retrain.sh
set -eu
cd "$(dirname "$0")/.."
PY=${PY:-python3}
stamp=$(date +%Y%m%d-%H%M)
mkdir -p "models/history/$stamp"
cp models/seat_d20.pt models/seat_d20.json models/activity_d20.pt models/activity_d20.json "models/history/$stamp/"
days=$(for d in recordings/*-day recordings/*-log; do if [ -f "$d/labels_d20.json" ]; then echo "$d"; fi; done)   # unlabelled days (e.g. today) are skipped
echo "seat model on: $days"
$PY -m training.train_seat $days
$PY -m training.train_activity recordings/activity --res 720
$PY -m training.check_reported recordings/*-log   # do the new models now get the reported mistakes right?
echo "done; old models in models/history/$stamp"
