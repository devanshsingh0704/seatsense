#!/usr/bin/env bash
# Office-hours start (desk-day-start.timer: 09:30 Mon-Sat, and 2 min after every boot):
# start the dashboard and the frame logger once the shared GPU has room, never outside 09:30-19:00.
set -u
NEED_MB=3000   # the dashboard's models need about 1.5-2 GB; keep headroom for the other projects
[ "$(date +%u)" = 7 ] && { echo "Sunday: office closed, not starting"; exit 0; }
while :; do
    now=$(date +%H%M)
    [ "$now" -ge 1900 ] && { echo "after 19:00: not starting today"; exit 0; }
    [ "$now" -lt 0930 ] && { sleep 60; continue; }
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "${free:-0}" -ge "$NEED_MB" ]; then
        systemctl --user start desk-presence.service desk-frame-logger.service
        echo "started at $(date +%H:%M) with ${free} MB GPU memory free"; exit 0
    fi
    echo "$(date +%H:%M) GPU has ${free:-?} MB free (< ${NEED_MB}): waiting 5 min"
    sleep 300
done
