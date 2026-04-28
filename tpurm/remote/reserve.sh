#!/usr/bin/env bash
set -euo pipefail

: "${ACTION:?}"

PID_FILE=/tmp/tpurm_reserve.pid
LOG_FILE=/tmp/tpurm_reserve.log
STATUS_FILE=/tmp/tpurm_reserve.status

remove_stale_libtpu_lockfile() {
  holder_pids="$(sudo lsof -t -w /dev/accel* /dev/vfio/* 2>/dev/null | sort -u || true)"
  if [ -n "$holder_pids" ]; then
    echo "$(date '+[%y-%m-%d %H:%M:%S]') lockfile cleanup skipped: TPU holders still present: $holder_pids" >> "$LOG_FILE"
    return 0
  fi
  if [ -e /tmp/libtpu_lockfile ]; then
    sudo rm -f /tmp/libtpu_lockfile
    echo "$(date '+[%y-%m-%d %H:%M:%S]') removed stale /tmp/libtpu_lockfile" >> "$LOG_FILE"
  fi
}

start() {
  if [ -f "$PID_FILE" ]; then
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" >/dev/null 2>&1; then
      exit 0
    fi
    rm -f "$PID_FILE"
  fi

  remove_stale_libtpu_lockfile

  python_bin="$(command -v python3 || command -v python || true)"
  if [ -z "$python_bin" ]; then
    echo "$(date '+[%y-%m-%d %H:%M:%S]') failed: python not found" > "$STATUS_FILE"
    echo "$(date '+[%y-%m-%d %H:%M:%S]') [reserve] ERROR: python not found" >> "$LOG_FILE"
    exit 1
  fi

  echo "$(date '+[%y-%m-%d %H:%M:%S]') starting" > "$STATUS_FILE"
  nohup setsid "$python_bin" -u - <<'PY' >>"$LOG_FILE" 2>&1 &
import importlib
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback

STATUS_FILE = "/tmp/tpurm_reserve.status"
PYTHON_TARGET = "/tmp/tpurm_reserve_python"
JAX_RELEASES = "https://storage.googleapis.com/jax-releases/libtpu_releases.html"

running = True


def log(message):
    print(time.strftime("[%y-%m-%d %H:%M:%S]"), "[reserve]", message, flush=True)


def status(message):
    with open(STATUS_FILE, "w") as f:
        f.write(time.strftime("[%y-%m-%d %H:%M:%S]") + f" {message}\n")


def handle_term(signum, frame):
    global running
    running = False
    status("stopping")
    log(f"received signal {signum}, stopping")


signal.signal(signal.SIGTERM, handle_term)
signal.signal(signal.SIGINT, handle_term)


def run(cmd):
    log("$ " + shlex.join(cmd))
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.stdout:
        print(proc.stdout, end="", flush=True)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def ensure_pip():
    try:
        run([sys.executable, "-m", "pip", "--version"])
        return
    except Exception:
        log("pip is unavailable; trying ensurepip")
    run([sys.executable, "-m", "ensurepip", "--upgrade"])


def clear_jax_modules():
    for name in list(sys.modules):
        if name == "jax" or name.startswith("jax.") or name == "jaxlib" or name.startswith("jaxlib."):
            del sys.modules[name]


def target_has_jax():
    return (
        os.path.isdir(os.path.join(PYTHON_TARGET, "jax"))
        and os.path.isdir(os.path.join(PYTHON_TARGET, "jaxlib"))
    )


def jax_importable():
    if PYTHON_TARGET not in sys.path:
        sys.path.insert(0, PYTHON_TARGET)
    importlib.invalidate_caches()
    clear_jax_modules()
    try:
        importlib.import_module("jax")
        importlib.import_module("jaxlib")
        return True
    except Exception:
        return False


def ensure_jax():
    os.makedirs(PYTHON_TARGET, exist_ok=True)
    if target_has_jax() and jax_importable():
        return

    status("installing jax tpu holder")
    ensure_pip()
    try:
        run([
            sys.executable,
            "-m",
            "pip",
            "install",
            "--target",
            PYTHON_TARGET,
            "-U",
            "jax[tpu]",
            "-f",
            JAX_RELEASES,
        ])
    except Exception:
        log("JAX install failed; removing partial target")
        shutil.rmtree(PYTHON_TARGET, ignore_errors=True)
        raise

    if not jax_importable():
        log("JAX install finished, but jax/jaxlib still cannot be imported")
        clear_jax_modules()
        importlib.import_module("jax")
        importlib.import_module("jaxlib")


log(f"reservation daemon started with {sys.executable}")
status("bootstrapping")

while running:
    try:
        ensure_jax()

        import jax
        import jax.numpy as jnp

        devices = [device for device in jax.local_devices() if device.platform == "tpu"]
        if not devices:
            status("failed: no local TPU devices visible, retrying")
            log("no local TPU devices visible; retrying in 10s")
            time.sleep(10)
            continue

        status(f"holding {len(devices)} local TPU device(s)")
        log(f"holding {len(devices)} local TPU device(s): {devices}")
        xs = [
            jax.device_put(jnp.ones((1024, 1024), dtype=jnp.bfloat16), device=device)
            for device in devices
        ]

        while running:
            for x in xs:
                (x @ x).block_until_ready()
            time.sleep(0.1)
    except Exception:
        if not running:
            break
        status("failed: holder exception, retrying")
        log("ERROR while trying to hold TPU:")
        traceback.print_exc()
        time.sleep(10)

status("stopped")
log("reservation daemon stopped")
PY

  echo $! > "$PID_FILE"
}

stop() {
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [ -z "$pid" ] || ! kill -0 "$pid" >/dev/null 2>&1; then
    rm -f "$PID_FILE"
    remove_stale_libtpu_lockfile
    echo "$(date '+[%y-%m-%d %H:%M:%S]') stopped" > "$STATUS_FILE"
    exit 0
  fi

  kill -TERM "$pid" >/dev/null 2>&1 || true
  kill -TERM "-$pid" >/dev/null 2>&1 || true
  for _ in $(seq 1 10); do
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      rm -f "$PID_FILE"
      remove_stale_libtpu_lockfile
      echo "$(date '+[%y-%m-%d %H:%M:%S]') stopped" > "$STATUS_FILE"
      exit 0
    fi
    sleep 1
  done

  kill -KILL "$pid" >/dev/null 2>&1 || true
  kill -KILL "-$pid" >/dev/null 2>&1 || true
  rm -f "$PID_FILE"
  remove_stale_libtpu_lockfile
  echo "$(date '+[%y-%m-%d %H:%M:%S]') stopped" > "$STATUS_FILE"
}

status_cmd() {
  echo "status:"
  cat "$STATUS_FILE" 2>/dev/null || true
  echo "pid:"
  cat "$PID_FILE" 2>/dev/null || true
  echo "log tail:"
  tail -80 "$LOG_FILE" 2>/dev/null || true
}

case "$ACTION" in
  start) start ;;
  stop) stop ;;
  status) status_cmd ;;
  *) echo "unknown ACTION=$ACTION" >&2; exit 2 ;;
esac
