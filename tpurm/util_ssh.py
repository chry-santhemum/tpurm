import json
import shlex
import subprocess
import textwrap
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast, get_args

from .globals import DatasetName, NFS_SSD_US, NFS_US, REPO_ROOT
from .tpu import name_to_tpu
from .util_log import LogContext, run_cmd

def ensure_ssh_key(log_ctx: LogContext) -> None:
    key_path = Path.home() / ".ssh" / "google_compute_engine"
    if key_path.exists():
        return
    key_path.parent.mkdir(mode=0o700, exist_ok=True)
    log_ctx.log("SSH key not found, generating new key")
    run_cmd(
        ["ssh-keygen", "-t", "rsa", "-f", str(key_path), "-N", "", "-q"],
        log_ctx=log_ctx,
        check=True,
    )

@dataclass(slots=True)
class SSHResult:
    returncode: int  # -1 = retry exhausted
    stdout: str = ""
    log_dir: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0
    @property
    def retry_exhausted(self) -> bool:
        return self.returncode == -1

def ssh_log_dir(operation: str, tpu_name: str) -> Path:
    log_id = f"{time.strftime('%y%m%d%H%M%S')}_{tpu_name}_{uuid.uuid4().hex[:6]}"
    log_dir = REPO_ROOT / ".tpurm" / "logs" / "remote" / operation / log_id
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir

def gcloud_ssh(
    tpu_name: str, zone: str, command: str, *,
    operation: str,
    worker: str,
    timeout: float|None,
    max_ssh_tries: int,
    capture_output: bool,
    log_ctx: LogContext,
) -> SSHResult:  # type: ignore
    """Run a command on a TPU VM and capture the output."""
    ensure_ssh_key(log_ctx)
    cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "ssh", tpu_name,
        f"--zone={zone}",
        f"--worker={worker}",
        "--command", command,
    ]
    flat_cmd = shlex.join(cmd)
    log_ctx.log(f"$ {flat_cmd[:300]}")

    # Logging specific to this command should go to log_dir, not log_ctx
    log_dir = ssh_log_dir(operation, tpu_name)
    log_ctx.log(f"Logging to: {log_dir}")
    (log_dir / "gcloud_cmd.txt").write_text(flat_cmd)

    captured: list[str] = []

    def finish(returncode: int) -> SSHResult:
        result = SSHResult(
            returncode=returncode,
            stdout="".join(captured),
            log_dir=str(log_dir),
        )
        (log_dir / "result.json").write_text(
            json.dumps({
                "returncode": result.returncode,
                "log_dir": result.log_dir,
                "ended_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }, indent=4)
        )
        return result

    assert max_ssh_tries >= 1
    for attempt in range(1, max_ssh_tries + 1):
        retry_reason = None
        try:
            proc = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=timeout
            )
            output = proc.stdout
            returncode = proc.returncode
        except subprocess.TimeoutExpired as exc:
            output: str = exc.stdout or ""  # type: ignore
            returncode = -1
            retry_reason = f"timed out after {timeout}s"

        (log_dir / f"attempt_{attempt}.log").write_text(output)
        if capture_output and output:
            captured.append(output)
        if returncode == 0:
            return finish(0)
        if returncode == 255:
            retry_reason = "failed with return code 255"
        if retry_reason is None:
            return finish(returncode)
        if attempt < max_ssh_tries:
            log_ctx.log(f"SSH command {retry_reason} (attempt {attempt}/{max_ssh_tries}); retrying...")
            continue

        log_ctx.log(f"SSH retry limit reached ({max_ssh_tries}). See logs: {log_dir}")
        return finish(-1)


# Helpers for checking if a TPU is ready to launch jobs

_SETUP_MARKER = "__TPURM_SETUP__"

class SetupStatus(TypedDict):
    env: bool
    requirements: bool
    datasets: list[DatasetName]

def _parse_setup_line(line: str) -> dict[str, str] | None:
    if not line.startswith(_SETUP_MARKER + "\t"):
        return None
    out: dict[str, str] = {}
    for item in line.split("\t")[1:]:
        if "=" not in item:
            return None
        key, value = item.split("=", 1)
        out[key] = value
    return out

def check_setup(
    tpu_name: str, zone: str, *,
    log_ctx: LogContext,
    requirements_hash: str | None = None,
    timeout: float = 15,
    max_ssh_tries: int = 3,
) -> SetupStatus | None:
    """
    Check base environment, requirements stamp, and dataset mounts on all workers.
    Returns None if the SSH call failed.
    """
    log_ctx.log(f"Checking setup on {tpu_name}...")
    dataset_names = cast(tuple[DatasetName, ...], get_args(DatasetName))
    dataset_checks = "\n".join(
        f'[ -d /mnt/atticusw/data/{name} ] && datasets="${{datasets:+$datasets,}}{name}"'
        for name in dataset_names
    )
    script = textwrap.dedent(
        f"""\
        set -eu
        wid="${{TPU_WORKER_ID:-$(hostname)}}"
        env=1
        mountpoint -q {shlex.quote(NFS_US)} || env=0
        mountpoint -q {shlex.quote(NFS_SSD_US)} || env=0
        python3.13 -m pip --version >/dev/null 2>&1 || env=0
        requirements=1
        if [ -n "${{REQUIREMENTS_HASH:-}}" ]; then
          stamp="$HOME/.cache/tpurm/requirements.lock.sha"
          if [ ! -f "$stamp" ] || [ "$(cat "$stamp")" != "$REQUIREMENTS_HASH" ]; then
            requirements=0
          fi
        fi
        datasets=""
        {dataset_checks}
        printf '{_SETUP_MARKER}\\tworker=%s\\tenv=%s\\trequirements=%s\\tdatasets=%s\\n' \\
          "$wid" "$env" "$requirements" "$datasets"
        """
    )
    env_prefix = ""
    if requirements_hash is not None:
        env_prefix = f"REQUIREMENTS_HASH={shlex.quote(requirements_hash)}"
    remote_cmd = f"{env_prefix} bash -s <<'REMOTE_SCRIPT'\n{script}\nREMOTE_SCRIPT"
    result = gcloud_ssh(
        tpu_name, zone, remote_cmd,
        operation="check_setup",
        worker="all",
        log_ctx=log_ctx,
        timeout=timeout,
        max_ssh_tries=max_ssh_tries,
        capture_output=True,
    )
    if not result.ok:
        log_ctx.log(f"SSH failed on {tpu_name}: {result}")
        return None

    by_worker: dict[str, SetupStatus] = {}
    for line in result.stdout.splitlines():
        fields = _parse_setup_line(line.strip())
        if fields is None or not fields.get("worker"):
            continue
        worker = fields["worker"]
        try:
            worker_datasets = [
                cast(DatasetName, name)
                for name in fields.get("datasets", "").split(",")
                if name in dataset_names
            ]
            by_worker[worker] = {
                "env": fields["env"] == "1",
                "requirements": fields["requirements"] == "1",
                "datasets": worker_datasets,
            }
        except KeyError:
            continue

    if not by_worker:
        log_ctx.log("Setup check failed: no parseable markers returned")
        return None

    env_ok = all(info["env"] for info in by_worker.values())
    requirements_ok = all(info["requirements"] for info in by_worker.values())
    datasets = [
        name
        for name in dataset_names
        if all(name in info["datasets"] for info in by_worker.values())
    ]

    if not env_ok:
        failed = ", ".join(worker for worker, info in by_worker.items() if not info["env"])
        log_ctx.log(f"Setup check: base environment missing on workers: {failed}")
    if requirements_hash is not None and not requirements_ok:
        failed = ", ".join(worker for worker, info in by_worker.items() if not info["requirements"])
        log_ctx.log(f"Setup check: requirements missing on workers: {failed}")
    if env_ok and (requirements_hash is None or requirements_ok):
        log_ctx.log("Setup check: Passed")

    return {
        "env": env_ok,
        "requirements": requirements_ok,
        "datasets": datasets,
    }

def check_vacancy(
    tpu_name: str, zone: str, *,
    log_ctx: LogContext,
    timeout: float = 15,
    max_ssh_tries: int = 3,
) -> tuple[bool, str] | None:
    """
    SSH into worker 0 and check if TPU is in use.
    `None` means the SSH call failed.
    """
    cmd = (
        "sudo lsof -w /dev/accel* /dev/vfio/* 2>/dev/null || true; "
        "echo ---MARKER---; "
        "if [ -e /tmp/libtpu_lockfile ]; then echo PRESENT; fi; "
        "echo ---MARKER---; "
        "pgrep -af 'python|pip|gsutil|gcloud|apt-get|dpkg|zstd' 2>/dev/null | grep -v MARKER | grep -v networkd-dispatcher | grep -v unattended-upgrade || true; "
        "echo ---MARKER---; "
        "cat /proc/loadavg 2>/dev/null || true"
    )
    result = gcloud_ssh(
        tpu_name, zone, cmd,
        log_ctx=log_ctx,
        operation="check_vacancy",
        worker="0",
        timeout=timeout,
        capture_output=True,
        max_ssh_tries=max_ssh_tries,
    )
    if not result.ok:
        log_ctx.log(f"SSH failed on {tpu_name}: {result}")
        return None

    parts = result.stdout.split("---MARKER---")
    raw_holder_output = parts[0].strip() if len(parts) > 0 else ""
    lockfile_output = parts[1].strip() if len(parts) > 1 else ""
    python_output = parts[2].strip() if len(parts) > 2 else ""
    load_output = parts[3].strip() if len(parts) > 3 else ""

    holder_lines = [line for line in raw_holder_output.splitlines() if "/dev/accel" in line or "/dev/vfio" in line]
    holder_output = "\n".join(holder_lines).strip()
    lockfile_present = lockfile_output == "PRESENT"

    tpu = name_to_tpu(tpu_name, zone)
    if tpu is not None and tpu.owner == "atticusw":
        if python_output:
            python_examples = "\n".join(python_output.splitlines()[:1])
            log_ctx.log(f"  {tpu_name}: python process example:\n{python_examples}")
        if holder_output:
            holder_examples = "\n".join(holder_output.splitlines()[:3])
            log_ctx.log(f"  {tpu_name}: TPU holder examples:\n{holder_examples}")
        if lockfile_present:
            log_ctx.log(f"  {tpu_name}: /tmp/libtpu_lockfile present")
        log_ctx.log(f"  {tpu_name}: load output:\n{load_output}")

    load_parts = load_output.split()
    try:
        load_1m = float(load_parts[0]) if len(load_parts) >= 1 else 999.0
        load_5m = float(load_parts[1]) if len(load_parts) >= 2 else 999.0
    except ValueError:
        load_1m = 999.0
        load_5m = 999.0

    vacant = (
        holder_output == ""
        and not lockfile_present
        and python_output == ""
        and (load_5m < 2.0 or load_1m < 1.0)
    )
    return vacant, load_output
