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

from .common import TPU, DatasetName, REPO_ROOT, NFS_SSD_US, thread_log, run_cmd, gcloud_ssh, read_remote_script

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


def kill_remote_processes(tpu_name: str, zone: str):
    thread_log(f"Killing REMOTE processes on {tpu_name}...")
    cmd = (
        # Graceful shutdown first, then force-kill stragglers
        "sudo pkill -15 python || true\n"
        "sleep 2\n"
        "sudo pkill -9 python || true\n"
        # Avoid broad `pkill -f` patterns here: they can match this cleanup shell itself.
        "sudo pkill -9 zstd || true\n"
        "sudo pkill -9 pv || true\n"
        "sudo pkill -9 tar || true\n"
        "sudo fuser -k 8476/tcp >/dev/null 2>&1 || true\n"
        "sudo fuser -k /dev/vfio/0 >/dev/null 2>&1 || true\n"
        "sudo rm -rf /tmp/libtpu_lockfile /tmp/tpu_logs /tmp/*tpu* || true\n"
        # Wait for coordination service port to be free
        "for i in $(seq 1 10); do\n"
        "  sudo fuser 8476/tcp >/dev/null 2>&1 || break\n"
        "  sleep 3\n"
        "done\n"
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
        else:
            thread_log(f"Failed to kill remote processes on {tpu_name}: exit code {result.returncode}")


def _dataset_warmup_block(tpu: TPU, datasets: list[DatasetName] | None) -> str:
    if not datasets:
        return "echo '[worker] no dataset warmup requested'"

    lines = []
    for dataset in datasets:
        gcs_subpath = "data/imagenet" if dataset == "imagenet" else "data/fineweb10B"
        gcs_prefix = f"{tpu.bucket}/{gcs_subpath}"
        clean_dest = "false" if dataset == "imagenet" else "true"
        lines.extend([
            f'echo "[worker] ensuring dataset {dataset}..."',
            (
                f'if ACTION=check DATASET={shlex.quote(dataset)} bash "$WARMUP_SCRIPT" | grep -q YES; then\n'
                f'  echo "[worker] dataset {dataset} already present"\n'
                'else\n'
                f'  ACTION=warmup DATASET={shlex.quote(dataset)} '
                f'GCS_PREFIX={shlex.quote(gcs_prefix)} BASE=imagenet FINEWEB10B_SUFFIX=bin '
                f'TMPFS_MOUNT=/mnt/atticusw TMPFS_SIZE=270G CLEAN_DEST={clean_dest} REMOUNT_ON_CLEAN_FAIL=true '
                'bash "$WARMUP_SCRIPT"\n'
                f'  if ! ACTION=check DATASET={shlex.quote(dataset)} bash "$WARMUP_SCRIPT" | grep -q YES; then\n'
                f'    echo "[worker] ERROR: dataset {dataset} still missing after warmup" >&2\n'
                '    exit 11\n'
                '  fi\n'
                'fi'
            ),
        ])
    return "\n".join(lines)


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
    launch_token = stage_dir.split("/")[-1].split("__")[-2]
    log_dir = stage_dir_to_log_dir(stage_dir)
    os.makedirs(log_dir, exist_ok=True)
    os.chmod(log_dir, 0o777)

    # Substitute placeholders in command
    real_command = command.format(
        log_dir=log_dir,
        run_name=run_name,
        project_name=project_name,
    )

    # Clean up leftover processes before launching (best-effort).
    kill_remote_processes(tpu.name, tpu.zone)
    
    # Build command for each worker
    # rsync from staging dir to local disk
    local_work_dir = f"~/{stage_dir_suffix}"
    workdir_block = textwrap.dedent(
        f"""\
        for i in $(seq 1 10); do ls {stage_dir}/ > /dev/null 2>&1 && break; echo 'Waiting for NFS...'; sleep 3; done
        mkdir -p {local_work_dir}
        rsync -a {stage_dir}/ {local_work_dir}/
        cd {local_work_dir}
        """
    ).strip()
    thread_log(f"Remote local workdir: {local_work_dir}")
    dataset_block = _dataset_warmup_block(tpu, datasets)
    if datasets:
        warmup_script_path = f"/tmp/tpurm_warmup_{launch_token}.sh"
        warmup_heredoc_tag = f"TPURM_WARMUP_{launch_token}"
        warmup_setup_block = textwrap.dedent(
            """\
            WARMUP_SCRIPT={warmup_script_path}
            cat > "$WARMUP_SCRIPT" <<'{warmup_heredoc_tag}'
            {warmup_script}
            {warmup_heredoc_tag}
            chmod +x "$WARMUP_SCRIPT"
            """
        ).format(
            warmup_script_path=shlex.quote(warmup_script_path),
            warmup_heredoc_tag=warmup_heredoc_tag,
            warmup_script=read_remote_script("warmup.sh"),
        ).strip()
    else:
        warmup_setup_block = "true"

    runner_script_path = f"~/{launch_token}.sh"
    runner_script = textwrap.dedent(
        """\
        #!/usr/bin/env bash
        set -eo pipefail
        export PATH="$HOME/.local/bin:$PATH"
        wandb login {wandb_key}

        export ZONE={zone}
        export SERVICE_ACCOUNT={service_account}

        WORKER_ID=${{TPU_WORKER_ID:-$(hostname)}}
        WORKER_LOG={log_dir}/worker_${{WORKER_ID}}.log
        EXIT_FILE={log_dir}/exit_${{WORKER_ID}}.{launch_token}

        trap 'rc=$?; echo "$rc" > "$EXIT_FILE"' EXIT

        {workdir_block}
        {warmup_setup_block}
        {dataset_block}

        set +e
        stdbuf -oL -eL {real_command} 2>&1 | tee "$WORKER_LOG"
        rc=$?
        set -e
        exit "$rc"
        """
    ).format(
        wandb_key=WANDB_KEY,
        zone=tpu.zone,
        service_account=tpu.service_account,
        log_dir=log_dir,
        launch_token=launch_token,
        workdir_block=workdir_block,
        warmup_setup_block=warmup_setup_block,
        dataset_block=dataset_block,
        real_command=real_command,
    )

    remote_cmd = textwrap.dedent(
        """\
        set -eo pipefail
        WORKER_ID=${{TPU_WORKER_ID:-$(hostname)}}
        cat > {runner_script_path} <<'TPURM_RUNNER'
        {runner_script}TPURM_RUNNER
        chmod +x {runner_script_path}
        nohup bash {runner_script_path} >/dev/null 2>&1 < /dev/null &
        echo $! > {log_dir}/pid_${{WORKER_ID}}.{launch_token}
        """
    ).format(
        log_dir=log_dir,
        runner_script_path=runner_script_path,
        runner_script=runner_script,
        launch_token=launch_token,
    )

    thread_log(f"Using service account: {tpu.service_account}")
    thread_log(f"Running at {tpu.name} {tpu.zone}")
    thread_log(f"Experiment logs at: {log_dir}", force_print=True)
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


def poll_launch(log_dir: str, launch_token: str, expected_workers: int) -> int | None:
    exit_files = sorted(Path(log_dir).glob(f"exit_*.{launch_token}"))
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
