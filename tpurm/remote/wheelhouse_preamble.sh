#!/usr/bin/env bash

# Shared preamble for wheelhouse build/install scripts.
# Sets up: worker ID detection, service account activation, wheelhouse dirs.

set -euo pipefail
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PATH="$HOME/.local/bin:$PATH"

: "${TAG:?}"
: "${SERVICE_ACCOUNT:?}"
: "${SA_KEY_FILE:=}"
: "${KEYS_DIR:?}"
: "${REGION:?}"

WORKER_HOST="$(hostname)"
if [[ "$WORKER_HOST" == *"-w-"* ]]; then
    WORKER_ID="${WORKER_HOST##*-w-}"
else
    WORKER_ID="$WORKER_HOST"
fi
echo "[wheelhouse][worker ${WORKER_ID}] host=${WORKER_HOST}"
setup_gcloud_auth "[wheelhouse][worker ${WORKER_ID}]"

WHEELHOUSE_DIR="${WHEELHOUSE_DIR:-/tmp/wheelhouse_${TAG}}"
WHEELHOUSE_TAR="${WHEELHOUSE_DIR}.tar.gz"

if [[ -n "$SERVICE_ACCOUNT" ]]; then
    echo "[wheelhouse][worker ${WORKER_ID}] service account: $SERVICE_ACCOUNT"
fi
