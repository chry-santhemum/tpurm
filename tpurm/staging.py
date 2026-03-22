import getpass
import random
import shlex
import shutil
import string
import subprocess
import textwrap
import time
from pathlib import Path, PurePosixPath

from .common import (
    REPO_ROOT, REMOTE_SCRIPTS_DIR, NFS_SSD_US,
    gcloud_ssh, run_cmd, thread_log,
)

RETENTION_DAYS = 7


def stage_dir_to_log_dir(stage_dir: str, root: str = NFS_SSD_US) -> str:
    stage_dir_suffix = stage_dir.split("/staging/")[-1]
    return str(Path(root) / "logs" / stage_dir_suffix)


def is_git_managed(path: Path) -> bool:
    """Checks if `path` is inside a git work tree."""
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def git_files_under(path: Path) -> list[str]:
    """
    Returns a list of files relative to `path`.
    Includes both tracked and untracked files, applying standard ignore rules.
    Empty dirs are not included.
    """
    out = subprocess.check_output(
        ["git", "-C", str(path), "ls-files", "-z", "-co", "--exclude-standard"],
    )
    return [p for p in out.decode().split("\0") if p]


def is_junk_path(path: Path) -> bool:
    """Files to ignore for non-git managed paths."""
    return (
        path.name == ".git"
        or
        "__pycache__" in path.parts
        or ".pytest_cache" in path.parts
        or ".mypy_cache" in path.parts
        or path.name == ".DS_Store"
        or path.suffix in {".pyc", ".pyo"}
    )


def walk(path: Path, stage_rel: PurePosixPath = PurePosixPath()) -> list[str]:
    """
    Returns a list of files relative to `path`.
    Works with symlinks and submodules.
    """
    if is_git_managed(path):
        out = []
        for rel_path in git_files_under(path):
            rel_path = rel_path.rstrip("/")
            child_path = path / rel_path
            child_stage_rel = stage_rel / PurePosixPath(rel_path)
            if child_path.is_dir():  # submodules or symlinks, I think
                out.extend(walk(child_path, child_stage_rel))
                continue
            out.append(child_stage_rel.as_posix())
        return out
    
    # If not git-managed, recurse over normal directories
    out = []
    for child in path.iterdir():
        child_stage_rel = stage_rel / child.name
        if is_junk_path(child):
            continue
        if child.is_dir():
            out.extend(walk(child, child_stage_rel))
            continue
        if child.is_file():
            out.append(child_stage_rel.as_posix())
    return out


def stage_code(run_name: str, project_name: str, retain=False, root=NFS_SSD_US) -> str:
    """
    rsync `REPO_ROOT` to the stage_dir.
    
    Returns:
        stage_dir: `{root}/staging/{user}/{project_name}/{run_name}__{launch_token}__{commit_id}`
    
    Args:
        run_name: Should be a timestamped unique identifier of the run.
        project_name: Only used to determine the staging subdirectory.
        retain: If True, the staged code will not be deleted after the retention period.
    """
    user = getpass.getuser()
    launch_token = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    commit_id = subprocess.check_output(
        ["git", "show", "-s", "--format=%h"],
        cwd=str(REPO_ROOT), text=True,
    ).strip()

    stage_dir = f"{root}/staging/{user}/{project_name}/{run_name}__{launch_token}__{commit_id}"
    Path(stage_dir).mkdir(parents=True, exist_ok=True)
    if retain:
        (Path(stage_dir) / ".retain").touch()
    
    # Get the files we want to stage
    thread_log(f"Staging files to {stage_dir}...")
    expanded_paths = list(dict.fromkeys(walk(REPO_ROOT)))
    files_from_input = "\0".join(expanded_paths) + ("\0" if expanded_paths else "")
    result = run_cmd(
        ["rsync", "-aL", "--from0", "--files-from=-", str(REPO_ROOT) + "/", stage_dir],
        input=files_from_input, text=True,
    )
    if result is None or result.returncode != 0:
        raise RuntimeError(f"rsync failed (exit code {result.returncode if result else 'N/A'})")

    # Also copy the remote shell scripts
    remote_stage_dir = Path(stage_dir, ".tpurm", "remote")
    remote_stage_dir.mkdir(parents=True, exist_ok=True)
    result = run_cmd(
        ["rsync", "-a", str(REMOTE_SCRIPTS_DIR) + "/", str(remote_stage_dir) + "/"],
        text=True,
    )
    if result is None or result.returncode != 0:
        raise RuntimeError(f"rsync failed (exit code {result.returncode if result else 'N/A'})")

    run_cmd(["chmod", "777", "-R", stage_dir])
    thread_log("Done staging.")

    thread_log("Cleaning up old staged code and logs...")
    staging_parent = f"{root}/staging/{getpass.getuser()}/{project_name}"
    cutoff = time.time() - RETENTION_DAYS * 86400
    for entry in Path(staging_parent).iterdir():
        if not entry.is_dir() or (entry / ".retain").exists():
            continue
        if entry.stat().st_mtime < cutoff:
            thread_log(f"  Removing {entry.name}")
            shutil.rmtree(entry, ignore_errors=True)
            shutil.rmtree(Path(stage_dir_to_log_dir(str(entry), root=root)), ignore_errors=True)
    thread_log("Done cleaning up.")

    return stage_dir


# Replace with kill.sh
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
        operation="kill_remote_processes",
        worker="all",
        timeout=60,
        capture_output=False,
        max_ssh_tries=2,
    )
    if not result.ok:
        if result.retry_exhausted:
            thread_log(f"Failed to kill remote processes on {tpu_name}: SSH retries exhausted (TPU likely preempted). Logs: {result.log_dir}")
            return True
        else:
            thread_log(f"Failed to kill remote processes on {tpu_name}: exit code {result.returncode}. Logs: {result.log_dir}")
            return False
    return True
