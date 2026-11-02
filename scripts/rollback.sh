#!/usr/bin/env bash
# Put back models saved in models/history/: the newest backup, or the one named. The models being
# replaced are kept in models/history/<now>-rolled-back/, so a rollback can itself be undone.
# Restarts the dashboard if it is running (outside office hours it loads them at the next start).
#   scripts/rollback.sh                   # the newest backup
#   scripts/rollback.sh 20261023-1342     # a named one (ls models/history)
set -eu
cd "$(dirname "$0")/.."
from=${1:-$(ls models/history | grep -v rolled-back | tail -1)}
[ -n "$from" ] && [ -d "models/history/$from" ] || { echo "no backup '$from' in models/history"; exit 1; }
keep=models/history/$(date +%Y%m%d-%H%M%S)-rolled-back
mkdir -p "$keep"
cp models/seat_d20.pt models/seat_d20.json models/activity_d20.pt models/activity_d20.json "$keep/"
cp "models/history/$from"/* models/
echo "models from $from are live again; the replaced ones are in $keep"
systemctl --user try-restart desk-presence
