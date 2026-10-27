#!/usr/bin/env bash
# Retrain the models on everything labelled so far. Runs on the GPU laptop when the GPU is free (it is
# shared). By hand only -- the Sunday job (tools/weekly.sh) puts the gate-tested candidate live instead. Keeps the
# current models in models/history/<stamp>/ so a worse day can be rolled back; live.py picks the new
# ones up on its next restart.
#   tools/retrain.sh                 # both
#   tools/retrain.sh seat            # or only the named ones: seat, activity
set -eu
cd "$(dirname "$0")/.."
PY=${PY:-python3}
what=" ${*:-seat activity} "
stamp=$(date +%Y%m%d-%H%M)
mkdir -p "models/history/$stamp"
cp models/seat_d20.pt models/seat_d20.json models/activity_d20.pt models/activity_d20.json "models/history/$stamp/"
days=$(for d in recordings/*-day recordings/*-log; do if [ -f "$d/labels_d20.json" ]; then echo "$d"; fi; done)   # unlabelled days (e.g. today) are skipped
echo "seat model on: $days"
case $what in *" seat "*) $PY tools/train_seat.py $days ;; esac
case $what in *" activity "*) $PY tools/train_activity.py recordings/activity --res 720 ;; esac
$PY tools/check_reported.py recordings/*-log   # do the new models now get the reported mistakes right?
echo "done; old models in models/history/$stamp"
