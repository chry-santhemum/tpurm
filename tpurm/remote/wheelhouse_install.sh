#!/usr/bin/env bash

# Install wheels from GCS wheelhouse tarball.
# Requires preamble: source wheelhouse_preamble.sh first (concatenated by Python).

: "${GCS_PREFIX:?}"

sudo rm -rf "$WHEELHOUSE_DIR" "$WHEELHOUSE_TAR"

echo "[wheelhouse][worker ${WORKER_ID}] downloading ${GCS_PREFIX}/wheelhouse_${TAG}_${REQUIREMENTS_HASH}.tar.gz"
gcloud storage cp "${GCS_PREFIX}/wheelhouse_${TAG}_${REQUIREMENTS_HASH}.tar.gz" "$WHEELHOUSE_TAR"
mkdir -p "$WHEELHOUSE_DIR"
tar -xzf "$WHEELHOUSE_TAR" -C "$WHEELHOUSE_DIR"

if [[ -f "$WHEELHOUSE_DIR/meta.txt" ]]; then
    echo "[wheelhouse][worker ${WORKER_ID}] meta:"
    cat "$WHEELHOUSE_DIR/meta.txt"
fi

python3.13 -m pip install --no-index --find-links "$WHEELHOUSE_DIR/wheels" \
    -r "$WHEELHOUSE_DIR/requirements.lock"
