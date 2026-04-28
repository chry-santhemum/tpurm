#!/usr/bin/env bash

# Check whether the wheelhouse tarball already exists in GCS.
# Requires preamble: source wheelhouse_preamble.sh first (concatenated by Python).

: "${GCS_PREFIX:?}"
: "${REQUIREMENTS_HASH:?}"

URI="${GCS_PREFIX}/wheelhouse_${TAG}_${REQUIREMENTS_HASH}.tar.gz"
echo "[wheelhouse][worker ${WORKER_ID}] checking ${URI}"

if gcloud storage ls "$URI" >/dev/null 2>&1; then
    echo "__TPURM_WHEELHOUSE_EXISTS__=1"
else
    echo "__TPURM_WHEELHOUSE_EXISTS__=0"
fi
