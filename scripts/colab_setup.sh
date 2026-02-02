#!/usr/bin/env bash
set -euo pipefail

# ====== user config ======
REPO_URL="${REPO_URL:-https://github.com/YOURNAME/YOURREPO.git}"
BRANCH="${BRANCH:-main}"

GCS_DATASET_URI="${GCS_DATASET_URI:-gs://YOUR_BUCKET/datasets/tartanair_depth.tar}"
GCS_INIT_CKPT_URI="${GCS_INIT_CKPT_URI:-gs://YOUR_BUCKET/weights/evggt.pt}"
GCS_OUT_URI="${GCS_OUT_URI:-gs://YOUR_BUCKET/experiments/vggt_run_$(date +%Y%m%d_%H%M%S)}"

WORKDIR="${WORKDIR:-/content/work}"
DATA_DIR="${DATA_DIR:-/content/data}"
OUT_DIR="${OUT_DIR:-/content/out}"
# =========================

mkdir -p "$WORKDIR" "$DATA_DIR" "$OUT_DIR"
cd "$WORKDIR"

# 1) clone
if [ ! -d repo ]; then
  git clone --branch "$BRANCH" "$REPO_URL" repo
fi
cd repo

# 2) deps (必要に応じて調整)
python -m pip install -U pip
pip install -r requirements.txt || true
# torchはColab環境に合わせる場合があるので、必要ならここで入れ直し
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3) gcloud / gsutil (Colabは大抵入ってるが無ければ)
if ! command -v gsutil >/dev/null 2>&1; then
  apt-get update
  apt-get install -y google-cloud-cli
fi

# 4) authenticate (手動ログイン方式：一番手軽)
# ここは初回にセル実行時に認証URLが出る
gcloud auth login --brief || true

# 5) download dataset + init weights
echo "[INFO] Download dataset from $GCS_DATASET_URI"
gsutil -m cp "$GCS_DATASET_URI" "$DATA_DIR/dataset.tar"

echo "[INFO] Extract dataset"
mkdir -p "$DATA_DIR/dataset"
tar -xf "$DATA_DIR/dataset.tar" -C "$DATA_DIR/dataset"

echo "[INFO] Download init ckpt from $GCS_INIT_CKPT_URI"
mkdir -p "$WORKDIR/weights"
gsutil -m cp "$GCS_INIT_CKPT_URI" "$WORKDIR/weights/evggt.pt"

# 6) prepare output and push a run meta
mkdir -p "$OUT_DIR"
echo "$GCS_OUT_URI" > "$OUT_DIR/GCS_OUT_URI.txt"
gsutil -m cp "$OUT_DIR/GCS_OUT_URI.txt" "$GCS_OUT_URI/GCS_OUT_URI.txt" || true

echo "[INFO] Ready. Example run:"
echo "python train.py --data_root $DATA_DIR/dataset --init_ckpt $WORKDIR/weights/evggt.pt --out_dir $OUT_DIR"