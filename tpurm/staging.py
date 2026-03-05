"""Stage code and launch jobs on a remote TPU VM."""

import getpass
import os
from pathlib import Path
import dotenv
import random
import string
import subprocess
import sys
import time
import textwrap

from .common import (
    _thread_local, TPU, REPO_ROOT, NFS_SSD_US,
    ensure_ssh_key, thread_log, run_cmd, gcloud_ssh
)
dotenv.load_dotenv(Path.home() / ".env")
WANDB_KEY = os.getenv("WANDB_KEY")


def stage_code(run_name: str, project_name: str, retain=False, stage_root_dir=NFS_SSD_US) -> str:
    """rsync current repo to the stage_dir, and return the stage_dir."""
    salt = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))

    try:
        commitid = subprocess.check_output(
            ["git", "show", "-s", "--format=%h"],
            cwd=str(REPO_ROOT), text=True,
        ).strip()
    except subprocess.CalledProcessError:
        commitid = "unkcommit"

    stage_dir = f"{stage_root_dir}/staging/{getpass.getuser()}/{project_name}/{run_name}-{salt}-{commitid}"

    thread_log(f"Staging files to {stage_dir}...")
    os.makedirs(stage_dir, exist_ok=True)
    if retain:
        Path(stage_dir, ".retain").touch()

    # Only copy files tracked by git + untracked-but-not-ignored files
    git_files = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, check=True,
    )
    result = run_cmd(
        ["rsync", "-a", "--files-from=-", str(REPO_ROOT) + "/", stage_dir],
        input=git_files.stdout, text=True,
    )
    if result is None or result.returncode != 0:
        print(f"ERROR: rsync failed (exit code {result.returncode if result else 'N/A'})", file=sys.stderr)
        sys.exit(1)
    run_cmd(["chmod", "777", "-R", stage_dir])
    thread_log("Done staging.")

    thread_log("Cleaning up old staged code...")
    staging_parent = f"{stage_root_dir}/staging/{getpass.getuser()}/{project_name}"
    cutoff = time.time() - 14 * 86400  # Default retention is 14 days
    for entry in Path(staging_parent).iterdir():
        if not entry.is_dir() or (entry / ".retain").exists():
            continue
        if entry.stat().st_mtime < cutoff:
            thread_log(f"  Removing {entry.name}")
            subprocess.run(["rm", "-rf", str(entry)], check=False)
    thread_log("Done cleaning up.")

    return stage_dir

def kill_remote_processes(tpu_name: str, zone: str):
    thread_log(f"Killing REMOTE processes on {tpu_name}...")
    cmd = (
        # Graceful shutdown first, then force-kill stragglers
        "sudo pkill -15 python || true\n"
        "sleep 2\n"
        "sudo pkill -9 python || true\n"
        "sudo fuser -k 8476/tcp >/dev/null 2>&1 || true\n"
        "sudo fuser -k /dev/vfio/0 >/dev/null 2>&1 || true\n"
        "sudo rm -rf /tmp/libtpu_lockfile /tmp/tpu_logs /tmp/*tpu* || true\n"
        # Wait for coordination service port to be free
        "for i in $(seq 1 10); do\n"
        "  sudo fuser 8476/tcp >/dev/null 2>&1 || break\n"
        "  sleep 1\n"
        "done\n"
    )
    gcloud_ssh(tpu_name, zone, cmd)

def launch(tpu: TPU, command: str, run_name: str, project_name: str, stage_dir: str) -> int:
    """
    Build and execute a command on all TPU VM workers.

    The command string may contain `{log_dir}`, `{run_name}`, `{project_name}`
    which will be replaced with the correct corresponding variables.
    """
    stage_dir_suffix = stage_dir.split("/staging/")[-1]
    log_dir = f"{NFS_SSD_US}/logs/{stage_dir_suffix}"
    os.makedirs(log_dir, exist_ok=True)
    os.chmod(log_dir, 0o777)

    # Substitute {log_dir} placeholder in command
    real_command = command.replace("{log_dir}", log_dir).replace("{run_name}", run_name).replace("{project_name}", project_name)

    # Clean up leftover processes before launching (runs from scheduler host,
    # so all workers are cleaned atomically before any worker starts the job).
    kill_remote_processes(tpu.name, tpu.zone)

    # Build remote command
    # Copy staged code from NFS to worker-local disk.
    local_work_dir = f"~/{stage_dir_suffix}"
    workdir_block = (
        f"for i in $(seq 1 10); do ls {stage_dir}/ > /dev/null 2>&1 && break; echo 'Waiting for NFS...'; sleep 3; done\n"
        f"mkdir -p {local_work_dir}\n"
        f"rsync -a {stage_dir}/ {local_work_dir}/\n"
        f"cd {local_work_dir}"
    )
    thread_log(f"Remote local workdir: {local_work_dir}")
    remote_cmd = textwrap.dedent(f"""
        set -eo pipefail
        export PATH="$HOME/.local/bin:$PATH"
        wandb login {WANDB_KEY}

        export ZONE={tpu.zone}
        export SERVICE_ACCOUNT={tpu.service_account}

        WORKER_ID=${{TPU_WORKER_ID:-$(hostname)}}
        WORKER_LOG=~/logs/{stage_dir_suffix}/worker_${{WORKER_ID}}.log
        mkdir -p $(dirname "$WORKER_LOG")

        {workdir_block}

        stdbuf -oL -eL {real_command} 2>&1 | tee "$WORKER_LOG"
    """).strip()

    thread_log(f"Using service account: {tpu.service_account}")
    thread_log(f"Running at {tpu.name} {tpu.zone}")
    thread_log(f"Experiment logs at: {log_dir}/output.log", force_print=True)

    result = gcloud_ssh(tpu.name, tpu.zone, remote_cmd, worker="all", max_ssh_retries=3)
    returncode = result.returncode if result is not None else EXIT_CODE_SSH_RETRY
    
    if returncode == EXIT_CODE_SSH_RETRY:
        thread_log("Job failed because SSH retries exceeded.")
        return returncode
    if returncode != 0:
        thread_log(f"Job exited with code {returncode}.")
        log_file = getattr(_thread_local, "log_file", None)
        if log_file and _has_fatal_python_error(log_file.name):
            thread_log("Fatal Python error detected.")
            return EXIT_CODE_FATAL
    return returncode


# Checking exit code
EXIT_CODE_FATAL = 2
EXIT_CODE_SSH_RETRY = -1
_FATAL_PATTERNS = [
    "ModuleNotFoundError:", "ImportError:", "SyntaxError:",
    "NameError:", "AttributeError:", "TypeError:",
    "KeyboardInterrupt",
]

def _has_fatal_python_error(log_path: str) -> bool:
    """Scan a launch log for fatal Python errors that won't be fixed by retrying."""
    try:
        with open(log_path) as f:
            for line in f:
                if any(pat in line for pat in _FATAL_PATTERNS):
                    return True
    except OSError:
        pass
    return False
