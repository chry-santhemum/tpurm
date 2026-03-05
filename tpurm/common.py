"""Shared constants and utils."""

import os
import re
import json
import shlex
import subprocess
import sys
import inspect
import threading
import time
import dotenv
from pathlib import Path
from typing import Any, Literal
from dataclasses import dataclass, field
from contextlib import contextmanager

dotenv.load_dotenv(Path.home() / ".env")
_thread_local = threading.local()

# TODO: change "size" to "type"

# Paths
SCRIPTS_DIR = Path(__file__).resolve().parent

def _resolve_repo_root() -> Path:
    env_root = os.environ.get("TPURM_REPO_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / ".git").exists():
            return candidate
    return cwd

REPO_ROOT = _resolve_repo_root()
NFS_US = "/kmh-nfs-us-mount"
NFS_SSD_US = "/kmh-nfs-ssd-us-mount"
DEFAULT_SA_KEY_FILE: str = os.getenv("DEFAULT_SA_KEY_FILE")  # type: ignore
DEFAULT_KEYS_DIR: str = os.getenv("DEFAULT_KEYS_DIR")  # type: ignore


# TPU configuration
# To add a new config: add service accounts, bucket, bucket key
TPU_CONFIGS: dict[str, dict[str, Any]] = {
    "v4": {
        "allowed_zones": ["us-central2-b"],
        "runtime_version": "tpu-ubuntu2204-base",
        "accelerator_type": lambda size: size,  # e.g. v4-32
    },
    "v5e": {
        "allowed_zones": ["us-west4-a", "us-central1-a"],
        "runtime_version": "v2-alpha-tpuv5-lite",
        "accelerator_type": lambda size: f"v5litepod-{size.split('-')[1]}",
    },
    "v5p": {
        "allowed_zones": ["us-central1-a", "us-east5-a"],
        "runtime_version": "v2-alpha-tpuv5",
        "accelerator_type": lambda size: size,
    },
    "v6e": {
        "allowed_zones": ["us-central1-b", "us-east5-b", "asia-northeast1-b"],
        "runtime_version": "v2-alpha-tpuv6e",
        "accelerator_type": lambda size: size,
    },
}

REGION_SERVICE_ACCOUNTS: dict[str, str] = {  # type: ignore
    "us-west4": os.getenv("REGION_SERVICE_ACCOUNTS_US_WEST4"),
    "us-east5": os.getenv("REGION_SERVICE_ACCOUNTS_US_EAST5"),
    "us-central1": os.getenv("REGION_SERVICE_ACCOUNTS_US_CENTRAL1"),
    "us-central2": os.getenv("REGION_SERVICE_ACCOUNTS_US_CENTRAL2"),
    "asia-northeast1": os.getenv("REGION_SERVICE_ACCOUNTS_ASIA_NORTHEAST1"),
}

REGION_BUCKETS = {
    "us-west4": "gs://kmh-gcp-us-west4",
    "us-east5": "gs://kmh-gcp-us-east5",
    "us-central1": "gs://kmh-gcp-us-central1",
    "us-central2": "gs://kmh-gcp-us-central2",
    "asia-northeast1": "gs://kmh-gcp-asia-northeast1-b",
}

AllocMode = Literal["spot", "preemptible", "persistent"]

ALLOCATION_MODES: dict[AllocMode, dict[str, str]] = {
    "spot": {
        "direct_flag": "--spot",
        "queued_flag": "--spot",
        "provisioning_model": "SPOT",
    },
    "preemptible": {
        "direct_flag": "--pre",
        "queued_flag": "--best-effort",
        "provisioning_model": "",
    },
    "persistent": {
        "direct_flag": "",
        "queued_flag": "",
        "provisioning_model": "",
    },
}

_LOG_MAX_BYTES = 5 * 1024 * 1024
_LOG_ROTATE_CHECK_EVERY = 1000

@dataclass
class TPU:
    """TPU properties known at creation."""
    size: str         # e.g. "v5p-64"
    mode: AllocMode   # e.g. "spot"
    owner: str        # e.g. "atticusw"
    id: str           # e.g. "260817"
    zone: str         # e.g. "us-central2-b"
    name: str = field(init=False)

    def __post_init__(self):
        self.name = f"kmh-tpuvm-{self.size}-{self.mode}-{self.owner}-{self.id}"
        self.family = size_to_family(self.size)
        self.region = zone_to_region(self.zone)
        self.wheelhouse_tag = self.family if self.family in ("v5p", "v6e") else ""
        self.config = TPU_CONFIGS[self.family]
        self.service_account: str = REGION_SERVICE_ACCOUNTS[self.region]
        self.bucket = REGION_BUCKETS[self.region]

        allowed_zones = self.config["allowed_zones"]
        if self.zone not in self.config["allowed_zones"]:
            print(f"Error: {self.family} support zones: {allowed_zones}. Requested: {self.zone}", file=sys.stderr)
            sys.exit(1)

def zone_to_region(zone: str) -> str:
    region, part = zone.rsplit("-", 1)
    assert len(part) == 1 and part.islower(), f"Invalid zone format: {zone}"
    return region

def size_to_family(tpu_size: str) -> str:
    prefix, _ = tpu_size.split("-")
    if prefix in ("v4", "v5e", "v5p", "v6e"):
        return prefix
    raise ValueError(f"Invalid TPU size: {tpu_size}. Expected v4-*, v5e-*, v5p-*, or v6e-*.")

def name_to_tpu(name: str, zone: str) -> TPU | None:
    """
    Parse a VM name like `kmh-tpuvm-v6e-64-spot-atticusw-12345` into a TPU.
    The only requirement is the `kmh-tpuvm-{family}-{chips}` prefix. Returns None if cannot be parsed.
    Mode defaults to `spot` when it can't be detected.
    The constructed TPU's `.name` is overridden with the actual VM name.
    """
    m = re.match(r"^kmh-tpuvm-(v4|v5e|v5p|v6e)-(\d+)(?:-(.*))?$", name)
    if not m:
        return None
    family, chips, remainder = m.groups()
    size = f"{family}-{chips}"
    remainder = remainder or ""

    # Try to strip a known mode from the front
    mode = "spot"
    for candidate in ("spot", "preemptible", "persistent"):
        if remainder == candidate:
            mode = candidate
            remainder = ""
            break
        if remainder.startswith(candidate + "-"):
            mode = candidate
            remainder = remainder[len(candidate) + 1:]
            break

    # Best-effort owner from what's left, rest is id (this is purely cosmetic)
    parts = remainder.split("-", 1) if remainder else []
    owner = parts[0] if parts else "unknown"
    tpu_id = parts[1] if len(parts) > 1 else ""

    tpu = TPU(size, mode, owner, tpu_id, zone)
    tpu.name = name  # override with actual VM name
    return tpu


@contextmanager
def set_thread_vars(**kwargs):
    for k, v in kwargs.items():
        setattr(_thread_local, k, v)
    try:
        yield
    finally:
        for k in kwargs:
            setattr(_thread_local, k, None)

def _maybe_rotate_log(f):
    try:
        pos = f.tell()
        if pos < _LOG_MAX_BYTES:
            return
        f.seek(0)
        data = f.read()
        keep = data[len(data) // 2:]
        # Find first newline to avoid partial line
        nl = keep.find("\n")
        if nl != -1:
            keep = keep[nl + 1:]
        f.seek(0)
        f.write(keep)
        f.truncate()
        f.flush()
    except Exception:
        pass


def thread_log(msg: str, force_print: bool=False):
    """
    Looks for thread-local log_file variable.
    If found (not None), writes to the file. Else, prints to stdout.
    """
    ts = time.strftime("%y-%m-%d %H:%M:%S")
    caller = inspect.currentframe().f_back.f_code.co_name  # type: ignore
    line = f"[{ts}] [{caller}]: {msg}"
    f = getattr(_thread_local, 'log_file', None)
    if f is not None:
        f.write(line + "\n")
        f.flush()
        _thread_local.log_write_count = getattr(_thread_local, 'log_write_count', 0) + 1
        if _thread_local.log_write_count % _LOG_ROTATE_CHECK_EVERY == 0:
            _maybe_rotate_log(f)
        if force_print:
            print(line, flush=True)
    else:
        print(line, flush=True)


# Shell execution helpers
# These functions assume thread-local vars have been set by the caller.

def run_cmd(cmd: list[str], **kwargs) -> subprocess.CompletedProcess | None:
    """
    Run a command in a subprocess, and return the completed process.
    Handles dry_run and logging redirection.

    Common kwargs:
        check: bool = False
        timeout: float | None = None
        capture_output: bool = False
    """
    flat = " ".join(cmd)
    thread_log(f"$ {flat}")
    if getattr(_thread_local, "dry_run", False):
        return None

    # Redirect subprocess stdout and stderr to log file
    f = getattr(_thread_local, "log_file", None)
    if f is not None and not kwargs.get("capture_output"):
        kwargs["text"] = True
        if "stdout" not in kwargs:
            kwargs["stdout"] = f
        if "stderr" not in kwargs:
            kwargs["stderr"] = f
        
    return subprocess.run(cmd, **kwargs)


def run_cmd_ssh_retry(cmd: list[str], max_retries: int) -> subprocess.CompletedProcess | None:
    """Run a subprocess, kill it after max_retries gcloud SSH retry messages."""
    flat = " ".join(cmd)
    thread_log(f"$ {flat}")
    if getattr(_thread_local, "dry_run", False):
        return None

    f = getattr(_thread_local, "log_file", None)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    retry_count = 0
    try:
        for line in proc.stdout:  # type: ignore
            if f:
                f.write(line)
                f.flush()
            if "Retrying:" in line:
                retry_count += 1
                if retry_count >= max_retries:
                    thread_log(f"SSH retried {max_retries} times, killing (TPU likely preempted)")
                    proc.kill()
                    proc.wait()
                    return subprocess.CompletedProcess(cmd, returncode=-1)
        proc.wait()
    except Exception:
        proc.kill()
        proc.wait()
        raise
    return subprocess.CompletedProcess(cmd, returncode=proc.returncode)


def ensure_ssh_key():
    key_path = Path.home() / ".ssh" / "google_compute_engine"
    if not key_path.exists():
        key_path.parent.mkdir(mode=0o700, exist_ok=True)
        thread_log("SSH key not found, generating new key")
        run_cmd(
            cmd=["ssh-keygen", "-t", "rsa", "-f", str(key_path), "-N", "", "-q"],
            check=True,
        )

def gcloud_ssh(
    tpu_name: str, zone: str, command: str, *,
    worker: str = "all", ssh_flags: list[str] | None = None,
    check: bool = False, timeout: float | None = None, capture_output: bool = False,
    max_ssh_retries: int | None = None,
) -> subprocess.CompletedProcess | None:
    """SSH into a TPU VM and run a command on the specified worker(s)."""
    ensure_ssh_key()
    cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "ssh", tpu_name,
        "--zone", zone,
        f"--worker={worker}",
        "--command", command,
    ]
    if ssh_flags:
        for flag in ssh_flags:
            cmd.extend(["--ssh-flag", flag])
    if max_ssh_retries is not None:
        return run_cmd_ssh_retry(cmd, max_ssh_retries)
    return run_cmd(cmd, check=check, timeout=timeout, capture_output=capture_output, text=capture_output)


def wait_for_ssh(tpu_name: str, zone: str, poll_interval: int = 15, max_attempts: int = 20) -> bool:
    """Poll until SSH works on all workers. Returns True if successful."""
    for attempt in range(1, max_attempts + 1):
        result = gcloud_ssh(
            tpu_name, zone, "true",
            worker="all", check=False, timeout=30,
            capture_output=True,
        )
        if getattr(_thread_local, "dry_run", False):
            return True
        if result is not None and result.returncode == 0:
            thread_log(f"SSH ready on {tpu_name} (attempt {attempt})")
            return True
        thread_log(f"SSH not ready on {tpu_name} (attempt {attempt}/{max_attempts}), retrying in {poll_interval}s...")
        time.sleep(poll_interval)
    thread_log(f"Error: SSH never became ready on {tpu_name}")
    return False


def read_remote_script(name: str) -> str:
    """Read and return the text of a script in scheduler/remote/."""
    return (SCRIPTS_DIR / "remote" / name).read_text()

def run_remote_script(
    tpu_name: str, zone: str, script_name: str, *,
    env: dict[str, str] | None = None,
    max_ssh_retries: int | None = None,
) -> bool:
    """Run a script in scheduler/remote/ on all workers."""
    script = read_remote_script(script_name)
    env_prefix = ""
    if env:
        env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + " "
    remote_cmd = f"{env_prefix}bash -s <<'REMOTE_SCRIPT'\n{script}\nREMOTE_SCRIPT"
    result = gcloud_ssh(tpu_name, zone, remote_cmd, worker="all", max_ssh_retries=max_ssh_retries)
    if getattr(_thread_local, "dry_run", False):
        return True
    return result is not None and result.returncode == 0


def gcloud_create_tpu(
    tpu_name: str, zone: str, accelerator_type: str, runtime_version: str, *,
    service_account: str = "", mode_flag: str = "",
) -> subprocess.CompletedProcess | None:
    """Create a TPU VM."""
    cmd = [
        "gcloud", "alpha", "compute", "tpus", "tpu-vm", "create", tpu_name,
        f"--zone={zone}",
        f"--accelerator-type={accelerator_type}",
        f"--version={runtime_version}",
    ]
    if service_account:
        cmd.append(f"--service-account={service_account}")
    if mode_flag:
        cmd.append(mode_flag)
    return run_cmd(cmd)

def gcloud_delete_tpu(tpu_name: str, zone: str) -> subprocess.CompletedProcess | None:
    cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "delete", tpu_name,
        f"--zone={zone}",
        "--quiet",
    ]
    return run_cmd(cmd)

def gcloud_describe_tpu(tpu_name: str, zone: str) -> tuple[bool, str | None]:
    """Return (exists, state) for TPU describe."""
    cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "describe",
        tpu_name, "--zone", zone, "--format=json",
    ]
    result = run_cmd(cmd, capture_output=True)
    if result is None or result.returncode != 0:
        return False, None
    try:
        info = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError):
        return True, None
    return True, info.get("state")


# Helpers for monitoring TPU status

def check_env(tpu_name: str, zone: str) -> bool:
    """Check if the software environment is correctly initialized on all workers."""
    thread_log(f"Checking environment on {tpu_name}...")
    if getattr(_thread_local, "dry_run", False):
        return True
    checks = [
        ("NFS mounts", "test -d /kmh-nfs-us-mount && test -d /kmh-nfs-ssd-us-mount"),
        ("JAX/Flax import", "python3.13 -c 'import jax; import flax' 2>/dev/null"),
    ]
    for name, cmd in checks:
        result = gcloud_ssh(tpu_name, zone, cmd, worker="all", check=False)
        if result is None or result.returncode != 0:
            thread_log(f"Environment check failed: {name}")
            return False
    thread_log("Environment check: Passed")
    return True


def check_data_mount(tpu_name: str, zone: str) -> bool:
    """
    Check if TPU has the mounted imagenet data directory.
    Returns False if SSH timed out.
    """
    try:
        result = gcloud_ssh(
            tpu_name, zone,
            "test -d /mnt/atticusw/data/imagenet && echo YES || echo NO",
            worker="0", timeout=30, capture_output=True,
        )
    except subprocess.TimeoutExpired:
        thread_log(f"  {tpu_name}: SSH timed out")
        return False
    if getattr(_thread_local, "dry_run", False):
        return True
    return (result is not None and result.returncode == 0 and "YES" in result.stdout)

def check_vacancy(tpu_name: str, zone: str, timeout: float = 30) -> dict:
    """
    SSH into worker 0 and check if TPU is in use.
    vacant=None means SSH failed.
    """
    info = {"name": tpu_name, "vacant": None, "load": ""}
    cmd = (
        "lsof /dev/accel* 2>/dev/null || true; "
        "echo ---MARKER---; "
        "pgrep -af 'python|pip|gsutil|gcloud|apt-get|dpkg|zstd' 2>/dev/null | grep -v MARKER | grep -v networkd-dispatcher | grep -v unattended-upgrade || true; "
        "echo ---MARKER---; "
        "cat /proc/loadavg 2>/dev/null || true"
    )
    try:
        result = gcloud_ssh(
            tpu_name, zone, cmd,
            worker="0", check=False, timeout=timeout,
            capture_output=True,
        )
    except subprocess.TimeoutExpired:
        thread_log(f"  {tpu_name}: SSH timed out")
        return info
    if result is None or result.returncode != 0:
        if not getattr(_thread_local, "dry_run", False):
            thread_log(f"  {tpu_name}: SSH failed")
        return info

    parts = result.stdout.split("---MARKER---")
    accel_output = parts[0].strip() if len(parts) > 0 else ""
    python_output = parts[1].strip() if len(parts) > 1 else ""
    load_output = parts[2].strip() if len(parts) > 2 else ""

    if python_output != "":
        thread_log(f"  {tpu_name}: python_output: {python_output}")

    info["load"] = load_output
    load_1m = float(load_output.split()[0]) if load_output else 999.0
    load_5m = float(load_output.split()[1]) if load_output else 999.0
    info["vacant"] = (accel_output == "") and (python_output == "") and (load_5m < 2.0 or load_1m < 1.0)
    return info

def list_tpus(zone: str) -> list[dict]:
    cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "list",
        f"--zone={zone}", "--format=json",
    ]
    result = run_cmd(cmd, capture_output=True, text=True)
    if getattr(_thread_local, "dry_run", False):
        return []
    if result is None or result.returncode != 0:
        thread_log(f"Warning: gcloud list failed for zone {zone}: {result.stderr.strip() if result else 'returned None'}")
        return []
    return json.loads(result.stdout)
