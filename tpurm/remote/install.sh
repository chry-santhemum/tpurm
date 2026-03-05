#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

: "${REQUIREMENTS_LOCK:?}"

python3.13 -m pip install -r "$REQUIREMENTS_LOCK" \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html \
    --retries 5 --timeout 120
