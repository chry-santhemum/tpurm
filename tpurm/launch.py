import os
import shlex
import textwrap
from pathlib import Path

from .globals import ENV_VARS, DatasetName, DEFAULT_KEYS_DIR
from .staging import stage_dir_to_log_dir
from .tpu import TPU
from .util_log import LogContext
from .util_ssh import gcloud_ssh


def launch(
    tpu: TPU,
    command: str,
    run_name: str,
    project_name: str,
    stage_dir: str,
    resume_from: str | None = None,
    datasets: list[DatasetName] | None = None,
    log_dir: str | None = None,
    *,
    log_ctx: LogContext,
) -> int:
    """
    Dispatch a command on all TPU VM workers and return immediately.

    The command string may contain the strings `{log_dir}`, `{run_name}`, or `{project_name}`
    which will be replaced with the corresponding arguments.
    """
    stage_dir_suffix = stage_dir.split("/staging/")[-1]
    if log_dir is None:
        log_dir = stage_dir_to_log_dir(stage_dir)
    os.makedirs(log_dir, exist_ok=True)
    os.chmod(log_dir, 0o777)

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
    log_ctx.log(f"Using service account: {tpu.service_account}")
    log_ctx.log(f"Running at {tpu.name} {tpu.zone}")
    log_ctx.log(f"Experiment logs at: {log_dir}", force_print=True)
    
    wandb_key: str = ENV_VARS["WANDB_KEY"]  # type: ignore
    launch_env = {
        "LOG_DIR": log_dir,
        "STAGE_DIR": stage_dir,
        "STAGE_DIR_SUFFIX": stage_dir_suffix,
        "TRAIN_COMMAND": real_command,
        "WANDB_KEY": wandb_key,
        "DATASETS": datasets_env,
        "TPURM_RESUME_FROM": resume_from or "",
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
        tpu.name,
        tpu.zone,
        remote_cmd,
        operation="launch",
        worker="all",
        timeout=None,
        capture_output=False,
        max_ssh_tries=3,
        log_ctx=log_ctx,
    )

    returncode = EXIT_CODE_SSH_RETRY if result.retry_exhausted else result.returncode
    if returncode != 0:
        log_ctx.log(f"Failed to launch job: exit code {returncode}. Logs: {result.log_dir}")
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
