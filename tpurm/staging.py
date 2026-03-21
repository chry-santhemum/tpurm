"""Stage code and launch jobs on a remote TPU VM."""
import getpass
import os
from pathlib import Path
import dotenv
import random
import shlex
import string
import subprocess
import sys
import textwrap
import time

from .common import (
    TPU, DatasetName, DEFAULT_KEYS_DIR, REPO_ROOT, REMOTE_SCRIPTS_DIR, NFS_SSD_US,
    gcloud_ssh, run_cmd, thread_log,
)

dotenv.load_dotenv(Path.home() / ".env")
WANDB_KEY = os.getenv("WANDB_KEY")
ALL_DATASETS = ("imagenet", "fineweb10B")


def stage_code(run_name: str, project_name: str, retain=False, stage_root_dir=NFS_SSD_US) -> str:
    """
    rsync current git repo to the stage_dir, and return the stage_dir.
    
    Args:
        run_name: Should be a timestamped unique identifier of the run.
        project_name: Purely cosmetic for this function.
        retain: If True, the staged code will not be deleted after the 7-day retention period.
    """
    launch_token = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    commit_id = subprocess.check_output(
        ["git", "show", "-s", "--format=%h"],
        cwd=str(REPO_ROOT), text=True,
    ).strip()

    stage_dir = f"{stage_root_dir}/staging/{getpass.getuser()}/{project_name}/{run_name}__{launch_token}__{commit_id}"
    thread_log(f"Staging files to {stage_dir}...")
    os.makedirs(stage_dir, exist_ok=True)
    if retain:
        Path(stage_dir, ".retain").touch()

    # Only copy files tracked by git + untracked-but-not-ignored files
    git_files = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, check=True,
    )
    raw_paths = [p for p in git_files.stdout.splitlines() if p]
    expanded_paths: list[str] = []
    seen: set[str] = set()
    for rel_path in raw_paths:
        src_path = REPO_ROOT / rel_path
        if src_path.is_symlink() and src_path.is_dir():
            prefix = rel_path.rstrip("/")
            for sub_path in src_path.rglob("*"):
                if not sub_path.is_file():
                    continue
                if "__pycache__" in sub_path.parts:
                    continue
                staged_subpath = f"{prefix}/{sub_path.relative_to(src_path).as_posix()}"
                if staged_subpath in seen:
                    continue
                seen.add(staged_subpath)
                expanded_paths.append(staged_subpath)
            continue

        if rel_path in seen:
            continue
        seen.add(rel_path)
        expanded_paths.append(rel_path)

    files_from_input = "\n".join(expanded_paths) + ("\n" if expanded_paths else "")
    result = run_cmd(
        ["rsync", "-aL", "--files-from=-", str(REPO_ROOT) + "/", stage_dir],  # materialize symlinked files
        input=files_from_input, text=True,
    )
    if result is None or result.returncode != 0:
        print(f"ERROR: rsync failed (exit code {result.returncode if result else 'N/A'})", file=sys.stderr)
        sys.exit(1)

    remote_stage_dir = Path(stage_dir, ".tpurm", "remote")
    remote_stage_dir.mkdir(parents=True, exist_ok=True)
    result = run_cmd(
        ["rsync", "-a", str(REMOTE_SCRIPTS_DIR) + "/", str(remote_stage_dir) + "/"],
        text=True,
    )
    if result is None or result.returncode != 0:
        print(f"ERROR: rsync failed (exit code {result.returncode if result else 'N/A'})", file=sys.stderr)
        sys.exit(1)

    run_cmd(["chmod", "777", "-R", stage_dir])
    thread_log("Done staging.")

    thread_log("Cleaning up old staged code...")
    staging_parent = f"{stage_root_dir}/staging/{getpass.getuser()}/{project_name}"
    cutoff = time.time() - 7 * 86400  # Default retention is 7 days
    for entry in Path(staging_parent).iterdir():
        if not entry.is_dir() or (entry / ".retain").exists():
            continue
        if entry.stat().st_mtime < cutoff:
            thread_log(f"  Removing {entry.name}")
            subprocess.run(["rm", "-rf", str(entry)], check=False)
    thread_log("Done cleaning up.")

    return stage_dir


def kill_remote_processes(tpu_name: str, zone: str, log_dir: str) -> bool:
    """Kill the TPURM-managed runner process group for one job."""
    thread_log(f"Killing REMOTE processes on {tpu_name}...")
    cmd = textwrap.dedent(
        f"""\
        set -euo pipefail
        WORKER_ID=${{TPU_WORKER_ID:-$(hostname)}}
        PID_FILE={shlex.quote(log_dir)}/pid_${{WORKER_ID}}.txt
        cleanup_tpu_runtime() {{
          holder_pids="$(sudo lsof -t -w /dev/accel* /dev/vfio/* 2>/dev/null | sort -u || true)"
          if [ -n "$holder_pids" ]; then
            sudo kill -TERM $holder_pids >/dev/null 2>&1 || true
            for i in $(seq 1 5); do
              holder_pids="$(sudo lsof -t -w /dev/accel* /dev/vfio/* 2>/dev/null | sort -u || true)"
              if [ -z "$holder_pids" ]; then
                break
              fi
              sleep 1
            done
            if [ -n "$holder_pids" ]; then
              sudo kill -KILL $holder_pids >/dev/null 2>&1 || true
              for i in $(seq 1 5); do
                holder_pids="$(sudo lsof -t -w /dev/accel* /dev/vfio/* 2>/dev/null | sort -u || true)"
                if [ -z "$holder_pids" ]; then
                  break
                fi
                sleep 1
              done
            fi
          fi
          holder_pids="$(sudo lsof -t -w /dev/accel* /dev/vfio/* 2>/dev/null | sort -u || true)"
          if [ -n "$holder_pids" ]; then
            return 1
          fi
          if [ -e /tmp/libtpu_lockfile ]; then
            sudo rm -f /tmp/libtpu_lockfile
          fi
        }}
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
        """
    )
    result = gcloud_ssh(
        tpu_name,
        zone,
        cmd,
        worker="all",
        timeout=60,
        capture_output=False,
        max_ssh_tries=2,
    )
    if not result.ok:
        if result.ssh_retry_exhausted:
            thread_log(f"Failed to kill remote processes on {tpu_name}: SSH retries exhausted (TPU likely preempted)")
            return True
        else:
            thread_log(f"Failed to kill remote processes on {tpu_name}: exit code {result.returncode}")
            return False
    return True


def stage_dir_to_log_dir(stage_dir: str) -> str:
    stage_dir_suffix = stage_dir.split("/staging/")[-1]
    return f"{NFS_SSD_US}/logs/{stage_dir_suffix}"

def launch(
    tpu: TPU,
    command: str,
    run_name: str,
    project_name: str,
    stage_dir: str,
    datasets: list[DatasetName] | None = None,
) -> int:
    """
    Dispatch a command on all TPU VM workers and return immediately.

    The command string may contain the strings `{log_dir}`, `{run_name}`, or `{project_name}`
    which will be replaced with the corresponding arguments.
    """
    stage_dir_suffix = stage_dir.split("/staging/")[-1]
    log_dir = stage_dir_to_log_dir(stage_dir)
    os.makedirs(log_dir, exist_ok=True)
    os.chmod(log_dir, 0o777)

    # Substitute placeholders in command
    real_command = command.format(
        log_dir=log_dir,
        run_name=run_name,
        project_name=project_name,
    )

    assert DEFAULT_KEYS_DIR, "DEFAULT_KEYS_DIR must be set"
    keys_dir = DEFAULT_KEYS_DIR
    sa_key_file = f"{keys_dir}/bucket-{tpu.region}.json"
    runner_script_path = f"{stage_dir}/.tpurm/remote/launch_runner.sh"
    datasets_env = " ".join(datasets or [])
    thread_log(f"Using service account: {tpu.service_account}")
    thread_log(f"Running at {tpu.name} {tpu.zone}")
    thread_log(f"Experiment logs at: {log_dir}", force_print=True)
    launch_env = {
        "LOG_DIR": log_dir,
        "STAGE_DIR": stage_dir,
        "STAGE_DIR_SUFFIX": stage_dir_suffix,
        "TRAIN_COMMAND": real_command,
        "WANDB_KEY": WANDB_KEY or "",
        "DATASETS": datasets_env,
        "TPU_BUCKET": tpu.bucket,
        "ZONE": tpu.zone,
        "SERVICE_ACCOUNT": tpu.service_account,
        "SA_KEY_FILE": sa_key_file,
        "KEYS_DIR": keys_dir,
        "REGION": tpu.region,
    }
    env_string = " ".join(f"{name}={shlex.quote(value)}" for name, value in launch_env.items())

    remote_cmd = textwrap.dedent(
        f"""\
        set -eo pipefail
        WORKER_ID=${{TPU_WORKER_ID:-$(hostname)}}
        PID_FILE={shlex.quote(log_dir)}/pid_${{WORKER_ID}}.txt
        EXIT_FILE={shlex.quote(log_dir)}/exit_${{WORKER_ID}}.txt
        RUNNER_SCRIPT={shlex.quote(runner_script_path)}
        for i in $(seq 1 10); do [ -f "$RUNNER_SCRIPT" ] && break; echo 'Waiting for NFS...'; sleep 3; done
        if [ ! -f "$RUNNER_SCRIPT" ]; then
          echo "runner script missing: $RUNNER_SCRIPT" >&2
          exit 1
        fi
        # SSH retries can partially dispatch workers, so launching must be idempotent.
        existing_pid="$(pgrep -u "$(id -u)" -f -x "bash $RUNNER_SCRIPT" | head -n1 || true)"
        if [ -n "$existing_pid" ]; then
          echo "$existing_pid" > "$PID_FILE"
          exit 0
        fi
        if [ -f "$EXIT_FILE" ]; then
          exit 0
        fi
        nohup setsid env {env_string} bash "$RUNNER_SCRIPT" >/dev/null 2>&1 < /dev/null &
        echo $! > "$PID_FILE"
        """
    )
    result = gcloud_ssh(
        tpu.name, tpu.zone, remote_cmd, 
        worker="all", 
        timeout=None, 
        capture_output=False, 
        max_ssh_tries=3
    )

    # This is the return code of the gcloud call, not the remote command
    if result.ssh_retry_exhausted:
        returncode = EXIT_CODE_SSH_RETRY
    else:
        returncode = result.returncode
    if returncode == EXIT_CODE_SSH_RETRY:
        thread_log(f"Failed to launch job: exit code {returncode} (SSH retries exceeded).")
    elif returncode != 0:
        thread_log(f"Failed to launch job: exit code {returncode}.")
    return returncode


def poll_launch(log_dir: str, expected_workers: int) -> int | None:
    exit_files = sorted(Path(log_dir).glob("exit_*.txt"))
    if len(exit_files) < expected_workers:
        return None

    final_rc = 0
    for p in exit_files:
        try:
            rc = int(p.read_text().strip().splitlines()[0])
        except (OSError, ValueError, IndexError):
            return EXIT_CODE_SSH_RETRY
        if rc != 0 and final_rc == 0:
            final_rc = rc
    return final_rc


# Exit code signaling
EXIT_CODE_FATAL = 2
EXIT_CODE_SSH_RETRY = -1
_FATAL_ERROR_PATTERNS = [
    "ModuleNotFoundError:", "ImportError:", "SyntaxError:",
    "NameError:", "AttributeError:", "TypeError:",
    "KeyboardInterrupt",
]

def has_fatal_error(log_path: str) -> bool:
    """Scan a launch log for fatal errors."""
    try:
        with open(log_path) as f:
            for line in f:
                if any(pat in line for pat in _FATAL_ERROR_PATTERNS):
                    return True
    except OSError:
        pass
    return False

def has_fatal_error_in_logs(log_dir: str) -> bool:
    for p in sorted(Path(log_dir).glob("worker_*.log")):
        if has_fatal_error(str(p)):
            return True
    return False
