#!/usr/bin/env bash
# Save one full-resolution D20 frame per clock minute during office hours (09:00-19:00), plus
# the live detector's per-desk answer at that moment if it is running, with how old that answer is
# (age, s) and whether the camera was online -- a frozen detector keeps serving its last answer, so
# training/daily.py treats an old or offline answer as missing. CPU only, no GPU.
# Feeds future labelling: recordings/<YYYYMMDD>-log/frames/d20_<second-of-day>.jpg + live_d20.jsonl
#   scripts/frame_logger.sh
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
    K="X-Service-Key: ${DASH_SERVICE_KEY:-}"   # the dashboard's sign-in lets this key read /state only
    timeout -k 2 8 sh -c "curl -sfk -m 3 -H '$K' https://127.0.0.1:8095/state || curl -sf -m 3 -H '$K' http://127.0.0.1:8095/state" | python3 -c "
import json, sys
s = json.load(sys.stdin); c = s['cameras'][0]
print(json.dumps({'t': $t, 'desks': {d['no']: [d['state'], d['p_sitting']] for d in c['desks']},
                  'age': s['det_age'], 'online': c['online']}))" \
        >> "$D/live_d20.jsonl" 2>/dev/null || true
done
