import json
import shlex
import subprocess
from typing import Any

from .util_log import run_cmd, LogContext

def gcloud_list(zone: str, *, log_ctx: LogContext) -> list[dict]:
    cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "list",
        f"--zone={zone}", "--format=json",
    ]
    log_ctx.log(f"$ {shlex.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log_ctx.log(f"Warning: gcloud list failed for zone {zone}: {result.stderr.strip()}")
        return []
    return json.loads(result.stdout)

def gcloud_create(
    tpu_name: str, zone: str, *,
    accelerator_type: str,
    runtime_version: str,
    service_account: str,
    mode_flag: str,
    log_ctx: LogContext
) -> subprocess.CompletedProcess:
    """Create a TPU VM."""
    cmd = [
        "gcloud", "alpha", "compute", "tpus", "tpu-vm", "create", tpu_name,
        f"--zone={zone}",
        f"--accelerator-type={accelerator_type}",
        f"--version={runtime_version}",
        f"--service-account={service_account}",
    ]
    if mode_flag:
        cmd.append(mode_flag)
    return run_cmd(cmd, log_ctx=log_ctx)

def gcloud_delete(
    tpu_name: str, zone: str, *,
    log_ctx: LogContext
) -> subprocess.CompletedProcess:
    cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "delete", tpu_name,
        f"--zone={zone}", "--quiet",
    ]
    return run_cmd(cmd, log_ctx=log_ctx)

def gcloud_describe(
    tpu_name: str, zone: str, *,
    log_ctx: LogContext
) -> dict[str, Any]|None:
    """
    JSON output:
    - len(info["networkEndpoints"]) is the number of workers
    - info["state"]
    - info["health"]
    """
    cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "describe", tpu_name,
        f"--zone={zone}", "--format=json",
    ]
    log_ctx.log(f"$ {shlex.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return
    try:
        info = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError):
        return

    log_ctx.log(f"TPU info: {info}")
    return info

def gcloud_storage_ls(path: str, *, log_ctx: LogContext) -> list[str]:
    cmd = ["gcloud", "storage", "ls", path]
    log_ctx.log(f"$ {shlex.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if stderr:
            log_ctx.log(f"Warning: gcloud storage ls failed for {path}: {stderr}")
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]
