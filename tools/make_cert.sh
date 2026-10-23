#!/usr/bin/env bash
# Self-made HTTPS certificate for the dashboard. Browsers show a one-time "not trusted" warning on
# each computer; after accepting it the connection is encrypted. The key stays in certs/ (not in git).
# With certs/ present, live.py serves HTTPS to the office network; without it, 127.0.0.1 only.
#   tools/make_cert.sh <this machine's office-network IP address>
set -eu
cd "$(dirname "$0")/.."
[ $# -eq 1 ] || { echo "usage: tools/make_cert.sh <ip address of this machine>"; exit 1; }
mkdir -p certs && chmod 700 certs
openssl req -x509 -newkey rsa:2048 -nodes -days 825 -subj "/CN=Desk Presence" \
    -addext "subjectAltName=IP:$1,IP:127.0.0.1,DNS:localhost" -keyout certs/key.pem -out certs/cert.pem 2>/dev/null
chmod 600 certs/key.pem
openssl x509 -in certs/cert.pem -noout -subject -enddate -ext subjectAltName
