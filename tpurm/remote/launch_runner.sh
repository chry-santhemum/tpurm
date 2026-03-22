#!/usr/bin/env bash

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./gcloud_auth.sh
source "$SCRIPT_DIR/gcloud_auth.sh"

: "${LOG_DIR:?}"
: "${STAGE_DIR:?}"
: "${STAGE_DIR_SUFFIX:?}"
: "${TRAIN_COMMAND:?}"
: "${TPU_BUCKET:?}"
: "${ZONE:?}"
: "${SERVICE_ACCOUNT:?}"
: "${SA_KEY_FILE:?}"
: "${KEYS_DIR:?}"
: "${REGION:?}"

WORKER_ID="${TPU_WORKER_ID:-$(hostname)}"
WORKER_LOG="${LOG_DIR}/worker_${WORKER_ID}.log"
BOOTSTRAP_LOG="${LOG_DIR}/bootstrap_${WORKER_ID}.log"
EXIT_FILE="${LOG_DIR}/exit_${WORKER_ID}.txt"
PID_FILE="${LOG_DIR}/pid_${WORKER_ID}.txt"
WARMUP_SCRIPT="${SCRIPT_DIR}/warmup.sh"
LOCAL_WORK_DIR="${HOME}/${STAGE_DIR_SUFFIX}"
REMOTE_WANDB_DIR="${HOME}/.cache/tpurm/wandb/${STAGE_DIR_SUFFIX}/${WORKER_ID}"
BOOTSTRAP_STAGE=bootstrap_start

log_bootstrap() {
  echo "$(date '+[%y-%m-%d %H:%M:%S]') [bootstrap] $*"
}

on_term() {
  log_bootstrap "SIGTERM stage=$BOOTSTRAP_STAGE pid=$$ ppid=$PPID"
  exit 143
}

on_exit() {
  rc=$?
  log_bootstrap "EXIT rc=$rc stage=$BOOTSTRAP_STAGE pid=$$ ppid=$PPID"
  ps -o pid,ppid,pgid,sid,stat,comm,args -p "$$" -p "$PPID" 2>/dev/null || true
  echo "$rc" > "$EXIT_FILE"
  rm -f "$PID_FILE"
}

exec >> "$BOOTSTRAP_LOG" 2>&1
trap on_term TERM
trap on_exit EXIT

export PATH="$HOME/.local/bin:$PATH"
export ZONE SERVICE_ACCOUNT SA_KEY_FILE KEYS_DIR REGION

setup_gcloud_auth "[bootstrap]"

if [ -n "${DATASETS:-}" ]; then
  WARMUP_LOG="${LOG_DIR}/warmup_${WORKER_ID}.log"
  BOOTSTRAP_STAGE=dataset_warmup
  log_bootstrap "starting $BOOTSTRAP_STAGE"
  (
    exec >> "$WARMUP_LOG" 2>&1
    for dataset in $DATASETS; do
      case "$dataset" in
        imagenet)
          GCS_PREFIX="${TPU_BUCKET}/data/imagenet"
          CLEAN_DEST=false
          ;;
        fineweb10B)
          GCS_PREFIX="${TPU_BUCKET}/data/fineweb10B"
          CLEAN_DEST=true
          ;;
        *)
          echo "[worker] ERROR: unsupported dataset '$dataset'" >&2
          exit 11
          ;;
      esac
      echo "[worker] ensuring dataset $dataset..."
      if ACTION=check DATASET="$dataset" bash "$WARMUP_SCRIPT" | grep -q YES; then
        echo "[worker] dataset $dataset already present"
        continue
      fi
      ACTION=warmup DATASET="$dataset" \
        GCS_PREFIX="$GCS_PREFIX" BASE=imagenet FINEWEB10B_SUFFIX=bin \
        TMPFS_MOUNT=/mnt/atticusw TMPFS_SIZE=270G CLEAN_DEST="$CLEAN_DEST" REMOUNT_ON_CLEAN_FAIL=true \
        bash "$WARMUP_SCRIPT"
      if ! ACTION=check DATASET="$dataset" bash "$WARMUP_SCRIPT" | grep -q YES; then
        echo "[worker] ERROR: dataset $dataset still missing after warmup" >&2
        exit 11
      fi
    done
  )
  log_bootstrap "finished $BOOTSTRAP_STAGE"
fi

BOOTSTRAP_STAGE=wandb_login
log_bootstrap "starting $BOOTSTRAP_STAGE"
mkdir -p "$REMOTE_WANDB_DIR"
export WANDB_DIR="$REMOTE_WANDB_DIR"
wandb login "$WANDB_KEY"
log_bootstrap "finished $BOOTSTRAP_STAGE"

BOOTSTRAP_STAGE=workdir_setup
log_bootstrap "starting $BOOTSTRAP_STAGE"
mkdir -p "$LOCAL_WORK_DIR"
rsync -a "$STAGE_DIR"/ "$LOCAL_WORK_DIR"/
cd "$LOCAL_WORK_DIR"
log_bootstrap "finished $BOOTSTRAP_STAGE"

BOOTSTRAP_STAGE=train_command
log_bootstrap "starting $BOOTSTRAP_STAGE"
# TPU images can leave stale driver logs owned by a previous user.
sudo mkdir -p /tmp/tpu_logs
sudo chown -R "$(id -un):$(id -gn)" /tmp/tpu_logs
sudo chmod 755 /tmp/tpu_logs
set +e
# Avoid a tee pipeline here so the wrapper exits when the command exits.
stdbuf -oL -eL bash -lc "$TRAIN_COMMAND" >> "$WORKER_LOG" 2>&1
rc=$?
set -e
log_bootstrap "finished $BOOTSTRAP_STAGE rc=$rc"
exit "$rc"
