#!/usr/bin/env bash

# Expects env vars set by caller:
#   ACTION=check|check_all|warmup
#   DATASET=imagenet|fineweb10B
#   GCS_PREFIX, BASE, FINEWEB10B_SUFFIX, TMPFS_MOUNT, TMPFS_SIZE, DEST,
#   CLEAN_DEST, REMOUNT_ON_CLEAN_FAIL

set -euo pipefail

ACTION="${ACTION:-warmup}"
DATASET="${DATASET:-imagenet}"
FINEWEB10B_SUFFIX="${FINEWEB10B_SUFFIX:-bin}"
TMPFS_MOUNT="${TMPFS_MOUNT:-/mnt/atticusw}"
TMPFS_SIZE="${TMPFS_SIZE:-270G}"
CLEAN_DEST="${CLEAN_DEST:-true}"
REMOUNT_ON_CLEAN_FAIL="${REMOUNT_ON_CLEAN_FAIL:-true}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-}"
SA_KEY_FILE="${SA_KEY_FILE:-}"
KEYS_DIR="${KEYS_DIR:-}"
REGION="${REGION:-}"

if [ "$ACTION" = "warmup" ]; then
  : "${SERVICE_ACCOUNT:?}"
  : "${KEYS_DIR:?}"
  : "${REGION:?}"
fi

if [ -z "$SA_KEY_FILE" ] && [ -n "$KEYS_DIR" ] && [ -n "$REGION" ]; then
  SA_KEY_FILE="${KEYS_DIR%/}/bucket-${REGION}.json"
fi


dataset_mount_path() {
  dataset_mount_path_for "$DATASET"
}


dataset_mount_path_for() {
  dataset_name="$1"
  case "$dataset_name" in
    imagenet)
      echo "/mnt/atticusw/data/imagenet"
      ;;
    fineweb10B)
      echo "/mnt/atticusw/data/fineweb10B"
      ;;
    *)
      echo "[worker] ERROR: unsupported dataset '$dataset_name'" >&2
      exit 2
      ;;
  esac
}


run_check() {
  TARGET="$(dataset_mount_path)"
  if test -d "$TARGET"; then
    echo YES
  else
    echo NO
  fi
}


run_check_all() {
  for dataset_name in imagenet fineweb10B; do
    target="$(dataset_mount_path_for "$dataset_name")"
    if test -d "$target"; then
      echo "__TPURM_DATASET_STATUS__ ${dataset_name}=1"
    else
      echo "__TPURM_DATASET_STATUS__ ${dataset_name}=0"
    fi
  done
}


run_body() {
  if [ "$ACTION" = "check" ]; then
    run_check
    return
  fi
  if [ "$ACTION" = "check_all" ]; then
    run_check_all
    return
  fi
  if [ "$ACTION" != "warmup" ]; then
    echo "[worker] ERROR: unsupported ACTION '$ACTION'" >&2
    exit 2
  fi

  if [ -z "${DEST:-}" ]; then
    if [ "$DATASET" = "imagenet" ]; then
      DEST="${TMPFS_MOUNT}/data"
    else
      DEST="${TMPFS_MOUNT}/data/fineweb10B"
    fi
  fi

  echo "[worker] $(hostname): preparing for $DATASET warmup..."

  # Ensure stream/decompression tools exist; skip apt entirely on already-initialized TPUs.
  missing_tools=()
  command -v pv >/dev/null 2>&1 || missing_tools+=("pv")
  command -v zstd >/dev/null 2>&1 || missing_tools+=("zstd")
  if [ "${#missing_tools[@]}" -gt 0 ]; then
    export DEBIAN_FRONTEND=noninteractive
    APT_RETRIES=${APT_RETRIES:-20}
    ret=1
    sudo dpkg --configure -a || true
    sudo apt-get -y update >/dev/null || true
    for attempt in $(seq 1 "$APT_RETRIES"); do
      echo "[worker] apt install attempt $attempt/$APT_RETRIES for ${missing_tools[*]}"
      sudo apt-get -y install "${missing_tools[@]}" >/dev/null && ret=0 || ret=$?
      if [ "$ret" -eq 0 ]; then
        break
      fi
      echo '[worker] apt install failed, cleaning stale locks and retrying...'
      for f in /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock /var/lib/apt/lists/lock; do
        pids="$(sudo lsof -t "$f" 2>/dev/null || true)"
        if [ -n "$pids" ]; then
          sudo kill -9 $pids >/dev/null 2>&1 || true
        fi
      done
      sudo rm -f /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock /var/lib/apt/lists/lock || true
      sudo dpkg --configure -a || true
      sleep 5
    done
    if [ "$ret" -ne 0 ]; then
      echo '[worker] ERROR: apt installation failed after retries' >&2
      exit 7
    fi
  fi

  # mount tmpfs
  MNT="${TMPFS_MOUNT}"
  if grep -qs " $MNT " /proc/mounts; then
    echo "[worker] tmpfs already mounted at $MNT"
    if [ -n "${TMPFS_SIZE}" ]; then
      sudo mount -o remount,size=${TMPFS_SIZE} "$MNT" || true
    fi
  else
    echo "[worker] mounting tmpfs at $MNT (size=${TMPFS_SIZE})"
    sudo mkdir -p "$MNT"
    sudo mount -t tmpfs -o size=${TMPFS_SIZE},mode=0755,uid=0,gid=0 tmpfs "$MNT"
  fi

  echo '[worker] free -h:'
  free -h || true
  echo "[worker] df -h $MNT:"
  df -h "$MNT" || true

  # prepare target dirs
  RAMROOT="${DEST}"
  if [ "${CLEAN_DEST}" = "true" ] || [ "${CLEAN_DEST}" = "1" ]; then
    echo "[worker] cleaning $RAMROOT"
    if ! sudo rm -rf "$RAMROOT"; then
      echo "[worker] WARNING: rm failed; directory may be busy"
      if [ "${REMOUNT_ON_CLEAN_FAIL}" = "true" ] || [ "${REMOUNT_ON_CLEAN_FAIL}" = "1" ]; then
        echo "[worker] remounting tmpfs at $MNT to force clean"
        sudo umount -l "$MNT" || true
        sudo mount -t tmpfs -o size=${TMPFS_SIZE},mode=0755,uid=0,gid=0 tmpfs "$MNT"
      fi
    fi
  fi
  mkdir -p "$RAMROOT"

  START=$(date +%s)
  if [ -n "$SA_KEY_FILE" ]; then
    if [ -f "$SA_KEY_FILE" ]; then
      export GOOGLE_APPLICATION_CREDENTIALS="$SA_KEY_FILE"
      current_account="$(gcloud config get-value account 2>/dev/null || true)"
      if [ -n "$SERVICE_ACCOUNT" ] && [ "$current_account" != "$SERVICE_ACCOUNT" ]; then
        echo "[worker] activating service account via $SA_KEY_FILE"
        gcloud auth activate-service-account --key-file="$SA_KEY_FILE"
      fi
    else
      echo "[worker] WARN: key file not found: $SA_KEY_FILE"
    fi
  fi
  if [ "$DATASET" = "imagenet" ]; then
    BASE="${BASE:-imagenet}"
    echo "[worker] listing parts: ${GCS_PREFIX}/${BASE}.tar*"
    LIST_RETRIES=5
    for i in $(seq 1 "$LIST_RETRIES"); do
      mapfile -t PARTS < <((gcloud storage ls "${GCS_PREFIX}/${BASE}.tar"* 2>/dev/null || true) | sort)
      if [ "${#PARTS[@]}" -gt 0 ]; then
        break
      fi
      echo "[worker] no parts found yet (try $i/$LIST_RETRIES). Retrying in 5s..."
      sleep 5
    done
    if [ "${#PARTS[@]}" -eq 0 ]; then
      echo '[worker] ERROR: no parts found after retries.' >&2
      exit 2
    fi
    printf '[worker] found %d parts\n' "${#PARTS[@]}"

    use_zstd=0
    for p in "${PARTS[@]}"; do
      if [[ "$p" == *.zst* ]]; then
        use_zstd=1
        break
      fi
    done

    if [ "$use_zstd" -eq 1 ]; then
      gcloud storage cat "${PARTS[@]}" | pv -ptebar | zstd -d -c | tar -C "$RAMROOT" -xf -
    else
      gcloud storage cat "${PARTS[@]}" | pv -ptebar | tar -C "$RAMROOT" -xf -
    fi
  elif [ "$DATASET" = "fineweb10B" ]; then
    if [ -z "${GCS_PREFIX:-}" ]; then
      echo "[worker] ERROR: GCS_PREFIX is required for fineweb10B warmup" >&2
      exit 2
    fi
    echo "[worker] downloading fineweb10B *.${FINEWEB10B_SUFFIX} files from ${GCS_PREFIX} to ${RAMROOT}"
    timeout 600s gcloud storage cp "${GCS_PREFIX}/*.${FINEWEB10B_SUFFIX}" "$RAMROOT/" && ret=0 || ret=$?
    if [ "$ret" -ne 0 ]; then
      echo "[worker] ERROR: gcloud storage cp failed with code $ret" >&2
      exit 27
    fi
    num_files=$(ls "$RAMROOT"/*."${FINEWEB10B_SUFFIX}" 2>/dev/null | wc -l)
    echo "[worker] downloaded ${num_files} fineweb10B files"
  else
    echo "[worker] ERROR: unsupported dataset '$DATASET'" >&2
    exit 2
  fi

  END=$(date +%s)
  echo "[worker] warmup done in $((END-START))s"
  chmod a+rX -R "$RAMROOT"
}


if [ "$EUID" -ne 0 ] && [ "$ACTION" = "warmup" ]; then
  echo "[worker] not running as root, rerunning with sudo..."
  exec sudo env \
    ACTION="$ACTION" DATASET="$DATASET" \
    GCS_PREFIX="${GCS_PREFIX:-}" BASE="${BASE:-}" FINEWEB10B_SUFFIX="$FINEWEB10B_SUFFIX" \
    TMPFS_MOUNT="$TMPFS_MOUNT" TMPFS_SIZE="$TMPFS_SIZE" \
    DEST="${DEST:-}" CLEAN_DEST="$CLEAN_DEST" \
    SERVICE_ACCOUNT="$SERVICE_ACCOUNT" SA_KEY_FILE="$SA_KEY_FILE" \
    KEYS_DIR="$KEYS_DIR" REGION="$REGION" \
    REMOUNT_ON_CLEAN_FAIL="$REMOUNT_ON_CLEAN_FAIL" \
    "$0"
fi

run_body
