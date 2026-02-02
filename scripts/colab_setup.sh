#!/usr/bin/env bash
set -euo pipefail

# This script assumes you already cloned the repo and are running:
#   %cd map-guided-vggt
#   !bash scripts/colab_setup.sh
#
# It will:
# - ensure gsutil exists
# - authenticate (interactive)
# - region-check (optional)
# - download dataset (supports folder of .tar or single .tar)
# - download init checkpoint
# - write OUT_DIR/GCS_OUT_URI.txt and create the output folder on GCS

# ====== user config ======
GCS_DATASET_URI="${GCS_DATASET_URI:-gs://train_80gb/tars_80gb}"               # folder OR file
GCS_INIT_CKPT_URI="${GCS_INIT_CKPT_URI:-gs://train_80gb/checkpoints/vggt_6ch.pt}"
GCS_OUT_URI="${GCS_OUT_URI:-gs://train_80gb/checkpoints/vggt_6ch_$(date +%Y%m%d_%H%M%S)}"

WORKDIR="${WORKDIR:-/content/work}"
DATA_DIR="${DATA_DIR:-/content/data}"
OUT_DIR="${OUT_DIR:-/content/out}"

BUCKET="${BUCKET:-train_80gb}"
SKIP_IF_BUCKET_US_AND_NOT_US="${SKIP_IF_BUCKET_US_AND_NOT_US:-1}"            # 1=skip download, 0=ignore
# =========================

mkdir -p "$WORKDIR" "$DATA_DIR" "$OUT_DIR"

echo "[INFO] Repo dir: $(pwd)"
echo "[INFO] WORKDIR=$WORKDIR"
echo "[INFO] DATA_DIR=$DATA_DIR"
echo "[INFO] OUT_DIR=$OUT_DIR"
echo "[INFO] GCS_DATASET_URI=$GCS_DATASET_URI"
echo "[INFO] GCS_INIT_CKPT_URI=$GCS_INIT_CKPT_URI"
echo "[INFO] GCS_OUT_URI=$GCS_OUT_URI"

# 1) deps (repo is already present)
python -m pip install -U pip
pip install -r requirements.txt || true

# 2) ensure gsutil exists
if ! command -v gsutil >/dev/null 2>&1; then
  echo "[INFO] Installing google-cloud-cli (gsutil)..."
  sudo apt-get update
  sudo apt-get install -y google-cloud-cli
fi

# 3) authenticate (interactive)
# (if already authenticated, this is a no-op)
gcloud auth login --brief || true

# 4) region check (optional)
if [[ "$SKIP_IF_BUCKET_US_AND_NOT_US" == "1" ]]; then
  BUCKET_LOC=$(gsutil ls -L -b "gs://$BUCKET" | grep "Location constraint" | awk '{print $3}' || true)
  echo "[INFO] Bucket location: ${BUCKET_LOC:-unknown}"

  ZONE=$(curl -s -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/zone || true)

  if [ -z "$ZONE" ]; then
    echo "[WARN] Could not detect Colab zone. Skip region check."
  else
    REGION=$(echo "$ZONE" | awk -F'/' '{print $NF}' | sed 's/-[a-z]$//')
    echo "[INFO] Colab region: $REGION"

    if [[ "$BUCKET_LOC" == "US" && "$REGION" != us-* ]]; then
      echo "[WARN] Bucket is US multi-region, but Colab is not in US."
      echo "[WARN] Skip dataset download and exit."
      exit 0
    fi
  fi
fi

# 5) download dataset
# Supports:
# - folder URI: gs://bucket/path   (will download *.tar)
# - file URI:   gs://bucket/path/file.tar
mkdir -p "$DATA_DIR/tars" "$DATA_DIR/dataset"

echo "[INFO] Download dataset from $GCS_DATASET_URI"
if [[ "$GCS_DATASET_URI" == *.tar ]]; then
  # single tar
  gsutil -m cp "$GCS_DATASET_URI" "$DATA_DIR/tars/"
else
  # folder/prefix containing multiple tars
  gsutil -m cp "${GCS_DATASET_URI%/}"/*.tar "$DATA_DIR/tars/"
fi

echo "[INFO] Extract dataset tars into $DATA_DIR/dataset"
shopt -s nullglob
for t in "$DATA_DIR/tars"/*.tar; do
  echo "[INFO]  extracting: $(basename "$t")"
  tar -xf "$t" -C "$DATA_DIR/dataset"
done
shopt -u nullglob

# 6) download init weights
echo "[INFO] Download init ckpt from $GCS_INIT_CKPT_URI"
mkdir -p "$WORKDIR/weights"
gsutil -m cp "$GCS_INIT_CKPT_URI" "$WORKDIR/weights/evggt.pt"

# 7) prepare output folder and write run meta
echo "$GCS_OUT_URI" > "$OUT_DIR/GCS_OUT_URI.txt"
gsutil -m cp "$OUT_DIR/GCS_OUT_URI.txt" "$GCS_OUT_URI/GCS_OUT_URI.txt" || true

echo "[INFO] Ready. Example run:"
echo "python train.py --data_root $DATA_DIR/dataset --init_ckpt $WORKDIR/weights/evggt.pt --out_dir $OUT_DIR"