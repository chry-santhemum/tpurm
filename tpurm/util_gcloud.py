import os
import json
import shlex
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .globals import DEFAULT_KEYS_DIR
from .tpu import REGION_BUCKETS
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

def _storage_key_file(path: str) -> Path | None:
    for region, bucket in REGION_BUCKETS.items():
        if path == bucket or path.startswith(bucket + "/"):
            return Path(DEFAULT_KEYS_DIR) / f"bucket-{region}.json"
    return None

def gcloud_storage_ls(path: str, *, log_ctx: LogContext) -> list[str]:
    cmd = ["gcloud", "storage", "ls", path]
    log_ctx.log(f"$ {shlex.join(cmd)}")
    key_file = _storage_key_file(path)
    if key_file is not None and key_file.is_file():
        with TemporaryDirectory(prefix="tpurm-gcloud-") as config_dir:
            env = os.environ.copy()
            env["CLOUDSDK_CONFIG"] = config_dir
            env["CLOUDSDK_CORE_DISABLE_PROMPTS"] = "1"
            env["CLOUDSDK_CORE_DISABLE_FILE_LOGGING"] = "1"
            env["GOOGLE_APPLICATION_CREDENTIALS"] = str(key_file)
            auth_cmd = [
                "gcloud", "auth", "activate-service-account",
                f"--key-file={key_file}", "--quiet",
            ]
            auth_result = subprocess.run(auth_cmd, capture_output=True, text=True, env=env)
            if auth_result.returncode != 0:
                stderr = auth_result.stderr.strip()
                if stderr:
                    log_ctx.log(f"Warning: gcloud auth failed for {path}: {stderr}")
                return []
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    else:
        result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if stderr:
            log_ctx.log(f"Warning: gcloud storage ls failed for {path}: {stderr}")
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]
