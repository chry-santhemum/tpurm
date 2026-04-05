#!/usr/bin/env bash
set -euo pipefail

: "${ACTION:?}"

PID_FILE=/tmp/tpurm_reserve.pid
LOG_FILE=/tmp/tpurm_reserve.log

start() {
  if [ -f "$PID_FILE" ]; then
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" >/dev/null 2>&1; then
      exit 0
    fi
    rm -f "$PID_FILE"
  fi

  nohup bash -lc '
    set -euo pipefail
    python_bin=""
    while [ -z "$python_bin" ]; do
      python_bin="$(command -v python3.13 || command -v python3 || true)"
      [ -n "$python_bin" ] || sleep 5
    done
    exec "$python_bin" -u - <<'"'"'PY'"'"'
import signal
import time

running = True

def handle_term(signum, frame):
    global running
    running = False

signal.signal(signal.SIGTERM, handle_term)
signal.signal(signal.SIGINT, handle_term)

x = None
while running:
    if x is None:
        try:
            import jax
            import jax.numpy as jnp

            devices = [d for d in jax.devices() if d.platform == "tpu"]
            if not devices:
                time.sleep(5)
                continue

            x = jax.device_put(
                jnp.ones((128, 128), dtype=jnp.bfloat16),
                device=devices[0],
            )
        except Exception:
            time.sleep(5)
            continue

    (x @ x).block_until_ready()
PY
  ' >>"$LOG_FILE" 2>&1 &

  echo $! > "$PID_FILE"
}

stop() {
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [ -z "$pid" ] || ! kill -0 "$pid" >/dev/null 2>&1; then
    rm -f "$PID_FILE"
    exit 0
  fi

  kill -TERM "$pid" >/dev/null 2>&1 || true
  for _ in $(seq 1 10); do
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      rm -f "$PID_FILE"
      exit 0
    fi
    sleep 1
  done

  kill -KILL "$pid" >/dev/null 2>&1 || true
  rm -f "$PID_FILE"
}

case "$ACTION" in
  start) start ;;
  stop) stop ;;
  *) echo "unknown ACTION=$ACTION" >&2; exit 2 ;;
esac
