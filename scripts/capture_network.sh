#!/usr/bin/env bash
set -euo pipefail

if ! command -v tcpdump >/dev/null 2>&1; then
  echo "tcpdump is required. Install it with your Linux package manager, then rerun." >&2
  exit 2
fi
mkdir -p evidence
OUT="evidence/network-monitor.txt"
echo "Capturing non-loopback traffic to $OUT"
echo "Start the app and run the offline demo in another terminal; stop this with Ctrl+C."
echo "Do not include private document content in the recorded output."
sudo tcpdump -i any -nn 'not net 127.0.0.0/8 and not host ::1' 2>&1 | tee "$OUT"
