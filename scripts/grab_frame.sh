#!/usr/bin/env bash
# One frame from an RTSP stream, saved whole or not at all: ffmpeg writes a temporary file that is
# renamed only if the grab succeeded and is not empty (16 Nov: a grab cut off by its timeout left a
# 0-byte frame, which crashed that evening's labelling set). Exit 1 = nothing saved.
#   scripts/grab_frame.sh <rtsp url> <out.jpg>
set -u
tmp="$(dirname "$2")/partial_$$.jpg"
if timeout -k 5 20 ffmpeg -loglevel error -y -rtsp_transport tcp -i "$1" -frames:v 1 -q:v 2 "$tmp" && [ -s "$tmp" ]; then
    mv "$tmp" "$2"
else
    rm -f "$tmp"
    exit 1
fi
