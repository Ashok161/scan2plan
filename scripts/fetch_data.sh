#!/usr/bin/env bash
# Unpack the raw benchmark captures (StrayScanner exports) into data/.
# Usage: scripts/fetch_data.sh [dir-containing-the-zips]   (default: repo root)
set -euo pipefail
cd "$(dirname "$0")/.."
SRC="${1:-.}"
mkdir -p data
for z in single_room.zip single_scan_floor_only.zip single_scan_with_ceiling.zip; do
  if [ -f "$SRC/$z" ]; then echo "unpacking $z"; unzip -q -o "$SRC/$z" -d data; else echo "missing $SRC/$z"; fi
done
ls data
