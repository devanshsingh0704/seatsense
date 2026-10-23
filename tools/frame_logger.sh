#!/usr/bin/env bash
# Save one full-resolution D20 frame per clock minute during office hours (09:00-19:00), plus
# the live detector's per-desk answer at that moment if it is running. CPU only, no GPU.
# Feeds future labelling: recordings/<YYYYMMDD>-log/frames/d20_<second-of-day>.jpg + live_d20.jsonl
#   tools/frame_logger.sh
set -u
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
URL="rtsp://$NVR_USER:$NVR_PASS@$NVR_HOST:554/cam/realmonitor?channel=20&subtype=0"
while :; do
    sleep $(( 60 - $(date +%s) % 60 ))                      # wake on the full minute
    h=$(date +%H); [ "$h" -ge 9 ] && [ "$h" -lt 19 ] || continue
    D=recordings/$(date +%Y%m%d)-log; mkdir -p "$D/frames"
    t=$(( $(date +%s) - $(date -d 00:00 +%s) ))
    timeout -k 5 20 ffmpeg -loglevel error -y -rtsp_transport tcp -i "$URL" -frames:v 1 -q:v 2 \
        "$D/frames/d20_$(printf %05d "$t").jpg" || echo "$(date +%T) frame grab failed"
    timeout -k 2 5 curl -s -m 3 127.0.0.1:8095/state | python3 -c "
import json, sys
c = json.load(sys.stdin)['cameras'][0]
print(json.dumps({'t': $t, 'desks': {d['no']: [d['state'], d['p_sitting']] for d in c['desks']}}))" \
        >> "$D/live_d20.jsonl" 2>/dev/null || true
done
