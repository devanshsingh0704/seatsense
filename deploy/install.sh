#!/usr/bin/env bash
# Install the office-hours schedule as user services on the GPU laptop (they survive reboots;
# needs `loginctl enable-linger` for the user, already on). Re-run after changing deploy/systemd/.
set -eu
cd "$(dirname "$0")/.."
mkdir -p ~/.config/systemd/user
cp deploy/systemd/*.service deploy/systemd/*.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now desk-day-start.timer desk-day-stop.timer desk-daily-review.timer
systemctl --user list-timers --no-pager | grep desk
