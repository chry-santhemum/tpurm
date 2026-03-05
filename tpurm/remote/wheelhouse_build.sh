#!/usr/bin/env bash

# Build wheelhouse tarball and upload to GCS.
# Requires preamble: source wheelhouse_preamble.sh first (concatenated by Python).

: "${GCS_PREFIX:?}"
: "${JAX_LINK:?}"
: "${REQUIREMENTS_LOCK:?}"

echo "[wheelhouse][worker ${WORKER_ID}] workdir: $WHEELHOUSE_DIR"
sudo rm -rf "$WHEELHOUSE_DIR" "$WHEELHOUSE_TAR"
mkdir -p "$WHEELHOUSE_DIR/wheels"

cp "$REQUIREMENTS_LOCK" "$WHEELHOUSE_DIR/requirements.lock"
python3.13 -m pip download -r "$REQUIREMENTS_LOCK" \
    -d "$WHEELHOUSE_DIR/wheels" -f "$JAX_LINK" \
    --retries 5 --timeout 120

{
    echo "tag=$TAG"
    echo "python=$(python3.13 -V 2>&1)"
    echo "pip=$(python3.13 -m pip --version)"
    echo "uname=$(uname -a)"
    echo "time_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$WHEELHOUSE_DIR/meta.txt"

tar -czf "$WHEELHOUSE_TAR" -C "$WHEELHOUSE_DIR" .
gcloud storage cp "$WHEELHOUSE_TAR" "${GCS_PREFIX}/wheelhouse_${TAG}_${REQUIREMENTS_HASH}.tar.gz"

echo "[wheelhouse][worker ${WORKER_ID}] uploaded to ${GCS_PREFIX}/wheelhouse_${TAG}_${REQUIREMENTS_HASH}.tar.gz"
