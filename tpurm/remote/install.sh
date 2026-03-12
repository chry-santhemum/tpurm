#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export PIP_DISABLE_PIP_VERSION_CHECK=1

: "${REQUIREMENTS_LOCK:?}"
: "${REQUIREMENTS_HASH:?}"

STAMP_DIR="$HOME/.cache/tpurm"
STAMP_FILE="$STAMP_DIR/requirements.lock.sha"
mkdir -p "$STAMP_DIR"

if [ -f "$STAMP_FILE" ] && [ "$(cat "$STAMP_FILE")" = "$REQUIREMENTS_HASH" ]; then
    echo "[install.sh] requirements already current"
    exit 0
fi

python3.13 -m pip install -r "$REQUIREMENTS_LOCK" \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html \
    --retries 5 --timeout 120

STAMP_TMP="$(mktemp)"
printf '%s\n' "$REQUIREMENTS_HASH" > "$STAMP_TMP"
mv "$STAMP_TMP" "$STAMP_FILE"
