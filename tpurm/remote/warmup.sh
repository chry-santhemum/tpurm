#!/usr/bin/env bash

# Expects env vars set by caller:
#   ACTION=check|check_all|warmup
#   DATASET=imagenet|fineweb
#   GCS_PREFIX, BASE, FINEWEB_SUFFIX, TMPFS_MOUNT, TMPFS_SIZE, DEST,
#   CLEAN_DEST, REMOUNT_ON_CLEAN_FAIL

set -euo pipefail

ACTION="${ACTION:-warmup}"
DATASET="${DATASET:-imagenet}"
FINEWEB_SUFFIX="${FINEWEB_SUFFIX:-bin}"
TMPFS_MOUNT="${TMPFS_MOUNT:-/mnt/atticusw}"
TMPFS_SIZE="${TMPFS_SIZE:-270G}"
CLEAN_DEST="${CLEAN_DEST:-true}"
REMOUNT_ON_CLEAN_FAIL="${REMOUNT_ON_CLEAN_FAIL:-true}"


dataset_mount_path() {
  dataset_mount_path_for "$DATASET"
}


dataset_mount_path_for() {
  dataset_name="$1"
  case "$dataset_name" in
    imagenet)
      echo "/mnt/atticusw/data/imagenet"
      ;;
    fineweb)
      echo "/mnt/atticusw/data/fineweb"
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
  for dataset_name in imagenet fineweb; do
    target="$(dataset_mount_path_for "$dataset_name")"
    if test -d "$target"; then
      echo "${dataset_name}=1"
    else
      echo "${dataset_name}=0"
    fi
  done
}


start_tpu_occupier() {
  cat > /tmp/tpurm_warmup_occupy.py <<'PY'
import jax
import jax.numpy as jnp

x = jnp.ones((1024, 1024), dtype=jnp.bfloat16)
y = jnp.ones((1024, 1024), dtype=jnp.bfloat16)
while True:
    (x @ y).block_until_ready()
    x, y = y, x
PY
  nohup python3.13 /tmp/tpurm_warmup_occupy.py >/tmp/tpurm_warmup_occupy.log 2>&1 < /dev/null &
  echo "$!" >/tmp/tpurm_warmup_occupy.pid
}


stop_tpu_occupier() {
  if [ -f /tmp/tpurm_warmup_occupy.pid ]; then
    kill -9 "$(cat /tmp/tpurm_warmup_occupy.pid)" >/dev/null 2>&1 || true
  fi
  pkill -f tpurm_warmup_occupy.py >/dev/null 2>&1 || true
  rm -f /tmp/tpurm_warmup_occupy.pid /tmp/tpurm_warmup_occupy.py
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
      DEST="${TMPFS_MOUNT}/data/fineweb"
    fi
  fi

  echo "[worker] $(hostname): preparing for $DATASET warmup..."

  # install dependencies with robust apt handling
  export DEBIAN_FRONTEND=noninteractive
  APT_RETRIES=${APT_RETRIES:-20}

  # Aggressively stop and disable apt auto-update services
  sudo systemctl stop apt-daily.timer apt-daily-upgrade.timer unattended-upgrades.service || true
  sudo systemctl disable apt-daily.timer apt-daily-upgrade.timer || true
  sudo systemctl mask unattended-upgrades.service apt-daily.service apt-daily-upgrade.service || true
  sudo pkill -9 unattended-upgrade || true
  sudo pkill -9 apt.systemd.daily || true

  # Clear stale locks and fix interrupted dpkg
  sudo rm -f /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock /var/lib/apt/lists/lock
  sudo dpkg --configure -a || true

  ret=1
  for attempt in $(seq 1 "$APT_RETRIES"); do
    echo "[worker] apt install attempt $attempt/$APT_RETRIES"
    sudo apt-get -y install zstd pv >/dev/null && ret=0 || ret=$?
    if [ "$ret" -eq 0 ]; then
      break
    fi
    echo '[worker] apt install failed, cleaning up and retrying...'
    (
      sudo systemctl stop unattended-upgrades || true
      sudo killall unattended-upgrade || true
      for f in /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock /var/lib/apt/lists/lock; do
        sudo kill -9 $(sudo lsof -t "$f" 2>/dev/null) 2>/dev/null || true
      done
      sudo rm -f /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock /var/lib/apt/lists/lock || true
      sudo dpkg --configure -a || true
    ) || true
    sleep 5
  done
  if [ "$ret" -ne 0 ]; then
    echo '[worker] ERROR: apt installation failed after retries' >&2
    exit 7
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

  start_tpu_occupier || true
  trap stop_tpu_occupier EXIT

  START=$(date +%s)
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
  elif [ "$DATASET" = "fineweb" ]; then
    if [ -z "${GCS_PREFIX:-}" ]; then
      echo "[worker] ERROR: GCS_PREFIX is required for fineweb warmup" >&2
      exit 2
    fi
    echo "[worker] downloading fineweb *.${FINEWEB_SUFFIX} files from ${GCS_PREFIX} to ${RAMROOT}"
    timeout 600s gcloud storage cp "${GCS_PREFIX}/*.${FINEWEB_SUFFIX}" "$RAMROOT/" && ret=0 || ret=$?
    if [ "$ret" -ne 0 ]; then
      echo "[worker] ERROR: gcloud storage cp failed with code $ret" >&2
      exit 27
    fi
    num_files=$(ls "$RAMROOT"/*."${FINEWEB_SUFFIX}" 2>/dev/null | wc -l)
    echo "[worker] downloaded ${num_files} fineweb files"
  else
    echo "[worker] ERROR: unsupported dataset '$DATASET'" >&2
    exit 2
  fi

  END=$(date +%s)
  echo "[worker] warmup done in $((END-START))s"
  chmod a+rX -R "$RAMROOT"
}


if [ "$EUID" -ne 0 ]; then
  echo "[worker] not running as root, rerunning with sudo..."
  sudo \
    ACTION="$ACTION" DATASET="$DATASET" \
    GCS_PREFIX="${GCS_PREFIX:-}" BASE="${BASE:-}" FINEWEB_SUFFIX="$FINEWEB_SUFFIX" \
    TMPFS_MOUNT="$TMPFS_MOUNT" TMPFS_SIZE="$TMPFS_SIZE" \
    DEST="${DEST:-}" CLEAN_DEST="$CLEAN_DEST" \
    REMOUNT_ON_CLEAN_FAIL="$REMOUNT_ON_CLEAN_FAIL" \
    bash -c "$(declare -f dataset_mount_path_for); $(declare -f dataset_mount_path); $(declare -f run_check); $(declare -f run_check_all); $(declare -f start_tpu_occupier); $(declare -f stop_tpu_occupier); $(declare -f run_body); run_body"
  exit $?
fi

run_body
