#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-/content/out}"
GCS_OUT_URI="${2:-}"

if [ -z "$GCS_OUT_URI" ]; then
  if [ -f "$OUT_DIR/GCS_OUT_URI.txt" ]; then
    GCS_OUT_URI="$(cat "$OUT_DIR/GCS_OUT_URI.txt")"
  else
    echo "GCS_OUT_URI not provided and not found in $OUT_DIR/GCS_OUT_URI.txt"
    exit 1
  fi
fi

echo "[INFO] Syncing $OUT_DIR -> $GCS_OUT_URI"
gsutil -m rsync -r "$OUT_DIR" "$GCS_OUT_URI"
