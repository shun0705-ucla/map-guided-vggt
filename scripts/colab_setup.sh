#!/usr/bin/env bash
set -euo pipefail

# ====== user config ======
GCS_DATASET_URI="${GCS_DATASET_URI:-gs://train_80gb/tars_80gb}"               
GCS_INIT_CKPT_URI="${GCS_INIT_CKPT_URI:-gs://train_80gb/checkpoints/vggt_6ch.pt}"
GCS_OUT_URI="${GCS_OUT_URI:-gs://train_80gb/checkpoints/vggt_6ch_$(date +%Y%m%d_%H%M%S)}"

WORKDIR="${WORKDIR:-/content/work}"
DATA_DIR="${DATA_DIR:-/content/data}"
OUT_DIR="${OUT_DIR:-/content/out}"

# Region guard
# Bucket is fixed: "us (multiple regions in United States)" => Location constraint: US
BUCKET_LOC_FIXED="${BUCKET_LOC_FIXED:-US}"

# If bucket is US multi-region, allow only Colab regions starting with "us-"
SKIP_IF_BUCKET_US_AND_NOT_US="${SKIP_IF_BUCKET_US_AND_NOT_US:-1}"  # 1=skip all heavy work, 0=ignore
# =========================

mkdir -p "$WORKDIR" "$DATA_DIR" "$OUT_DIR"

# 0) FAST region check (no gsutil/gcloud needed)
if [[ "$SKIP_IF_BUCKET_US_AND_NOT_US" == "1" && "$BUCKET_LOC_FIXED" == "US" ]]; then
  ZONE=$(curl -s -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/zone || true)

  if [ -z "$ZONE" ]; then
    echo "[WARN] Could not detect Colab zone. Skip fast region check."
  else
    REGION=$(echo "$ZONE" | awk -F'/' '{print $NF}' | sed 's/-[a-z]$//')
    echo "[INFO] Fast region check: bucket=$BUCKET_LOC_FIXED, colab_region=$REGION"
    if [[ "$REGION" != us-* ]]; then
      echo "[WARN] Bucket is US multi-region, but Colab is not in US."
      echo "[WARN] Skip setup/download and exit."
      exit 0
    fi
  fi
fi

echo "[INFO] Passed region check. Continue setup..."

# 1) deps (repo is already present)
python -m pip install -U pip
# delete torchaudio to avoid version conflict
pip uninstall -y torchaudio || true
# install specific torch+cuda version
pip install -U --force-reinstall \
  torch==2.3.1 torchvision==0.18.1 \
  --index-url https://download.pytorch.org/whl/cu121
# install other dependencies
pip install -r requirements_colab.txt

# 2) ensure gsutil exists
if ! command -v gsutil >/dev/null 2>&1; then
  echo "[INFO] Installing google-cloud-cli (gsutil)..."
  sudo apt-get update
  sudo apt-get install -y google-cloud-cli
fi

# 3) authenticate (interactive; required for private bucket)
gcloud auth login --brief || true

# 4) download dataset
mkdir -p "$DATA_DIR/tars" "$DATA_DIR/dataset"

echo "[INFO] Download dataset from $GCS_DATASET_URI"
if [[ "$GCS_DATASET_URI" == *.tar ]]; then
  gsutil -m cp "$GCS_DATASET_URI" "$DATA_DIR/tars/"
else
  gsutil -m cp "${GCS_DATASET_URI%/}"/*.tar "$DATA_DIR/tars/"
fi

echo "[INFO] Extract dataset tars into $DATA_DIR/dataset"
shopt -s nullglob
for t in "$DATA_DIR/tars"/*.tar; do
  echo "[INFO]  extracting: $(basename "$t")"
  tar -xf "$t" -C "$DATA_DIR/dataset"
done
shopt -u nullglob

# 5) download init weights
echo "[INFO] Download init ckpt from $GCS_INIT_CKPT_URI"
mkdir -p "$WORKDIR/weights"
gsutil -m cp "$GCS_INIT_CKPT_URI" "$WORKDIR/weights/evggt.pt"

# 6) prepare output folder and write run meta
echo "$GCS_OUT_URI" > "$OUT_DIR/GCS_OUT_URI.txt"
gsutil -m cp "$OUT_DIR/GCS_OUT_URI.txt" "$GCS_OUT_URI/GCS_OUT_URI.txt" || true

echo "[INFO] Ready. Example run:"
echo "python train.py --data_root $DATA_DIR/dataset --init_ckpt $WORKDIR/weights/evggt.pt --out_dir $OUT_DIR"