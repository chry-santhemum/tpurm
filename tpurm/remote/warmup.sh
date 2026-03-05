#!/usr/bin/env bash

# Expects env vars set by caller:
#   GCS_PREFIX, BASE, TMPFS_MOUNT, TMPFS_SIZE, DEST, CLEAN_DEST, REMOUNT_ON_CLEAN_FAIL

set -euo pipefail

run_body() {
  echo "[worker] $(hostname): preparing..."

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
  for attempt in $(seq 1 $APT_RETRIES); do
    echo "[worker] apt install attempt $attempt/$APT_RETRIES"
    sudo apt-get -y install zstd pv >/dev/null && ret=0 || ret=$?
    if [ "$ret" -eq 0 ]; then break; fi
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

  echo '[worker] free -h:'; free -h || true
  echo "[worker] df -h $MNT:"; df -h "$MNT" || true

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

  # list and sort GCS parts
  echo "[worker] listing parts: ${GCS_PREFIX}/${BASE}.tar*"
  LIST_RETRIES=5
  for i in $(seq 1 $LIST_RETRIES); do
    mapfile -t PARTS < <( (gcloud storage ls "${GCS_PREFIX}/${BASE}.tar"* 2>/dev/null || true) | sort )
    if [ "${#PARTS[@]}" -gt 0 ]; then break; fi
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
    if [[ "$p" == *.zst* ]]; then use_zstd=1; break; fi
  done

  # stream download -> decompress -> extract
  set -euo pipefail
  START=$(date +%s)
  if [ "$use_zstd" -eq 1 ]; then
    gcloud storage cat "${PARTS[@]}" | pv -ptebar | zstd -d -c | tar -C "$RAMROOT" -xf -
  else
    gcloud storage cat "${PARTS[@]}" | pv -ptebar | tar -C "$RAMROOT" -xf -
  fi
  END=$(date +%s)
  echo "[worker] extract done in $((END-START))s"
  chmod a+rX -R "$RAMROOT"
}

if [ "$EUID" -ne 0 ]; then
  echo "[worker] not running as root, rerunning with sudo..."
  sudo \
    GCS_PREFIX="$GCS_PREFIX" BASE="$BASE" \
    TMPFS_MOUNT="$TMPFS_MOUNT" TMPFS_SIZE="$TMPFS_SIZE" \
    DEST="$DEST" CLEAN_DEST="$CLEAN_DEST" \
    REMOUNT_ON_CLEAN_FAIL="$REMOUNT_ON_CLEAN_FAIL" \
    bash -c "$(declare -f run_body); run_body"
  exit $?
fi

run_body
