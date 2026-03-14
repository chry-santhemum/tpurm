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

if [[ -z "$SA_KEY_FILE" && -n "$KEYS_DIR" && -n "$REGION" ]]; then
    SA_KEY_FILE="${KEYS_DIR%/}/bucket-${REGION}.json"
fi
if [[ -n "$SA_KEY_FILE" ]]; then
    if [[ -f "$SA_KEY_FILE" ]]; then
        export GOOGLE_APPLICATION_CREDENTIALS="$SA_KEY_FILE"
        current_account="$(gcloud config get-value account 2>/dev/null || true)"
        if [[ "$current_account" != "$SERVICE_ACCOUNT" ]]; then
            echo "[wheelhouse][worker ${WORKER_ID}] activating service account via $SA_KEY_FILE"
            gcloud auth activate-service-account --key-file="$SA_KEY_FILE"
        fi
    else
        echo "[wheelhouse][worker ${WORKER_ID}] WARN: key file not found: $SA_KEY_FILE"
    fi
fi

WHEELHOUSE_DIR="${WHEELHOUSE_DIR:-/tmp/wheelhouse_${TAG}}"
WHEELHOUSE_TAR="${WHEELHOUSE_DIR}.tar.gz"

if [[ -n "$SERVICE_ACCOUNT" ]]; then
    echo "[wheelhouse][worker ${WORKER_ID}] service account: $SERVICE_ACCOUNT"
fi
