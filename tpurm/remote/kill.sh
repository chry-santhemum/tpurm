#!/usr/bin/env bash

set -euo pipefail

: "${LOG_DIR:?}"

WORKER_ID="${TPU_WORKER_ID:-$(hostname)}"
PID_FILE="${LOG_DIR}/pid_${WORKER_ID}.txt"

cleanup_tpu_runtime() {
  holder_pids="$(sudo lsof -t -w /dev/accel* /dev/vfio/* 2>/dev/null | sort -u || true)"
  if [ -n "$holder_pids" ]; then
    sudo kill -TERM $holder_pids >/dev/null 2>&1 || true
    for i in $(seq 1 5); do
      holder_pids="$(sudo lsof -t -w /dev/accel* /dev/vfio/* 2>/dev/null | sort -u || true)"
      [ -z "$holder_pids" ] && break
      sleep 1
    done
    if [ -n "$holder_pids" ]; then
      sudo kill -KILL $holder_pids >/dev/null 2>&1 || true
      for i in $(seq 1 5); do
        holder_pids="$(sudo lsof -t -w /dev/accel* /dev/vfio/* 2>/dev/null | sort -u || true)"
        [ -z "$holder_pids" ] && break
        sleep 1
      done
    fi
  fi
  holder_pids="$(sudo lsof -t -w /dev/accel* /dev/vfio/* 2>/dev/null | sort -u || true)"
  [ -z "$holder_pids" ] || return 1
  [ ! -e /tmp/libtpu_lockfile ] || sudo rm -f /tmp/libtpu_lockfile
}

if [ ! -f "$PID_FILE" ]; then
  cleanup_tpu_runtime
  exit 0
fi

pid="$(cat "$PID_FILE" 2>/dev/null || true)"
case "$pid" in
  ''|*[!0-9]*)
    rm -f "$PID_FILE"
    cleanup_tpu_runtime
    exit 0
    ;;
esac

sudo kill -TERM -- "-$pid" >/dev/null 2>&1 || true
for i in $(seq 1 10); do
  if ! sudo kill -0 -- "-$pid" >/dev/null 2>&1; then
    rm -f "$PID_FILE"
    cleanup_tpu_runtime
    exit 0
  fi
  sleep 1
done

sudo kill -KILL -- "-$pid" >/dev/null 2>&1 || true
for i in $(seq 1 5); do
  if ! sudo kill -0 -- "-$pid" >/dev/null 2>&1; then
    rm -f "$PID_FILE"
    cleanup_tpu_runtime
    exit 0
  fi
  sleep 1
done

exit 1
