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
from typing import Any, Literal, get_args
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
DatasetName = Literal["imagenet", "fineweb"]
SUPPORTED_DATASETS = list(get_args(DatasetName))

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
    num_workers: int|None = None

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
        log_path = getattr(f, "name", None)
        if not isinstance(log_path, str):
            return
        f.flush()
        while Path(log_path).stat().st_size >= _LOG_MAX_BYTES:
            with open(log_path, "r+") as rf:
                data = rf.read()
                if not data:
                    break
                keep = data[len(data) // 2:]
                # Find first newline to avoid partial line
                nl = keep.find("\n")
                if nl != -1:
                    keep = keep[nl + 1:]
                rf.seek(0)
                rf.write(keep)
                rf.truncate()
                rf.flush()

        # Keep append-mode handles writing from EOF after rotation.
        f.seek(0, os.SEEK_END)
    except Exception:
        pass


def _bump_log_rotation_check(f):
    _thread_local.log_write_count = getattr(_thread_local, "log_write_count", 0) + 1
    if _thread_local.log_write_count % _LOG_ROTATE_CHECK_EVERY == 0:
        _maybe_rotate_log(f)


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
        _bump_log_rotation_check(f)
        if force_print:
            print(line, flush=True)
    else:
        print(line, flush=True)


# Shell execution helpers
# These functions assume thread-local vars have been set by the caller.

def run_cmd(cmd: list[str], **kwargs) -> subprocess.CompletedProcess | None:
    """
    Run a command in a subprocess, and return the completed process.
    Handles logging redirection.

    Common kwargs:
        check: bool = False
        timeout: float | None = None
        capture_output: bool = False
    """
    flat = " ".join(cmd)
    thread_log(f"$ {repr(flat)[:200]}")

    # Redirect subprocess stdout and stderr to log file
    f = getattr(_thread_local, "log_file", None)
    redirected_to_log = False
    if f is not None and not kwargs.get("capture_output"):
        kwargs["text"] = True
        if "stdout" not in kwargs:
            kwargs["stdout"] = f
            redirected_to_log = True
        if "stderr" not in kwargs:
            kwargs["stderr"] = f
            redirected_to_log = True

    result = subprocess.run(cmd, **kwargs)
    if redirected_to_log:
        _maybe_rotate_log(f)
    return result


def ensure_ssh_key():
    key_path = Path.home() / ".ssh" / "google_compute_engine"
    if not key_path.exists():
        key_path.parent.mkdir(mode=0o700, exist_ok=True)
        thread_log("SSH key not found, generating new key")
        run_cmd(
            cmd=["ssh-keygen", "-t", "rsa", "-f", str(key_path), "-N", "", "-q"],
            check=True,
        )

SSHStatus = Literal["ok", "error", "ssh_retry_exhausted"]

@dataclass(slots=True)
class SSHResult:
    status: SSHStatus
    returncode: int
    stdout: str = ""
    stderr: str = ""
    @property
    def ok(self) -> bool:
        return self.status == "ok"
    @property
    def ssh_retry_exhausted(self) -> bool:
        return self.status == "ssh_retry_exhausted"


def gcloud_ssh(
    tpu_name: str, zone: str, command: str, *,
    worker: str,
    timeout: float|None,
    capture_output: bool,
    max_ssh_tries: int,
    ssh_flags: list[str] | None = None,
) -> SSHResult:
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
    thread_log(f"$ {repr(' '.join(cmd))[:200]}")

    ssh_log_path = REPO_ROOT / ".tpurm" / "logs" / "gcloud_ssh.log"
    ssh_log_path.parent.mkdir(parents=True, exist_ok=True)
    captured: list[str] = []
    retry_count = 0  # Number of "Retrying:" lines seen.
    retry_exhausted = False
    started_at = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        assert proc.stdout is not None
        with open(ssh_log_path, "a") as ssh_log:
            for line in proc.stdout:
                if capture_output:
                    captured.append(line)
                ssh_log.write(line)
                ssh_log.flush()
                _bump_log_rotation_check(ssh_log)

                if "Retrying:" in line:
                    retry_count += 1
                    if retry_count >= max_ssh_tries:
                        thread_log(
                            f"SSH attempt limit reached ({max_ssh_tries}); TPU likely preempted"
                        )
                        proc.kill()
                        retry_exhausted = True
                        break
                if timeout is not None and (time.monotonic() - started_at) > timeout:
                    raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout, output="".join(captured))
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    stdout = "".join(captured)
    if retry_exhausted:
        return SSHResult(
            status="ssh_retry_exhausted",
            returncode=-1,
            stdout=stdout,
        )

    returncode = proc.returncode if proc.returncode is not None else 1
    return SSHResult(
        status="ok" if returncode == 0 else "error",
        returncode=returncode,
        stdout=stdout,
    )


def read_remote_script(name: str) -> str:
    """Read and return the text of a script in scheduler/remote/."""
    return (SCRIPTS_DIR / "remote" / name).read_text()


# TODO: fix this?
def run_remote_script(
    tpu_name: str, zone: str, script_name: str, *,
    env: dict[str, str] | None = None,
    max_ssh_tries: int = 3,
) -> bool:
    """Run a script in scheduler/remote/ on all workers."""
    script = read_remote_script(script_name)
    env_prefix = ""
    if env:
        env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + " "
    remote_cmd = f"{env_prefix}bash -s <<'REMOTE_SCRIPT'\n{script}\nREMOTE_SCRIPT"
    result = gcloud_ssh(
        tpu_name, zone, remote_cmd, 
        worker="all", 
        timeout=None,
        capture_output=False,
        max_ssh_tries=max_ssh_tries
    )
    if result.ssh_retry_exhausted:
        thread_log(f"SSH retries exhausted while running {script_name} on {tpu_name} (TPU likely preempted)")
    return result.ok


def list_tpus(zone: str) -> list[dict]:
    cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "list",
        f"--zone={zone}", "--format=json",
    ]
    result = run_cmd(cmd, capture_output=True, text=True)
    if result is None or result.returncode != 0:
        thread_log(f"Warning: gcloud list failed for zone {zone}: {result.stderr.strip() if result else 'returned None'}")
        return []
    return json.loads(result.stdout)
    
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

def gcloud_describe_tpu(tpu_name: str, zone: str) -> tuple[bool, str | None, str | None, int | None]:
    """Return (exists, state, health, num_workers) for TPU describe."""
    cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "describe",
        tpu_name, "--zone", zone, "--format=json",
    ]
    result = run_cmd(cmd, capture_output=True)
    if result is None or result.returncode != 0:
        return False, None, None, None
    try:
        info = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError):
        return True, None, None, None

    num_workers = None
    endpoints = info.get("networkEndpoints")
    if isinstance(endpoints, list):
        num_workers = max(len(endpoints), 1)
    thread_log(f"Info: {info}")
    return True, info.get("state"), info.get("health"), num_workers


# Helpers for monitoring TPU status
# We make the convention that None means "SSH failed"

def check_env(tpu_name: str, zone: str, timeout: float = 15, max_ssh_tries: int=3) -> bool | None:
    """Check if the software environment is correctly initialized on all workers."""
    thread_log(f"Checking environment on {tpu_name}...")
    cmd = (
        'wid="${TPU_WORKER_ID:-$(hostname)}"; '
        "mounts=1; test -d /kmh-nfs-us-mount && test -d /kmh-nfs-ssd-us-mount || mounts=0; "
        "jax_flax=1; python3.13 -c 'import jax; import flax' >/dev/null 2>&1 || jax_flax=0; "
        "ok=$((mounts && jax_flax)); "
        'echo "__TPURM_ENV__ worker=${wid} mounts=${mounts} jax_flax=${jax_flax} ok=${ok}"; '
        "exit 0"
    )
    try:
        result = gcloud_ssh(
            tpu_name, zone, cmd,
            worker="all",
            timeout=timeout,
            capture_output=True,
            max_ssh_tries=max_ssh_tries,
        )
    except subprocess.TimeoutExpired:
        thread_log(f"SSH timed out: {tpu_name}")
        return
    if not result.ok:
        thread_log(f"SSH failed on {tpu_name}: {result}")
        return

    by_worker: dict[str, tuple[int, int, int]] = {}
    for line in result.stdout.splitlines():
        parsed = re.search(
            r"__TPURM_ENV__ worker=([^ ]+) mounts=([01]) jax_flax=([01]) ok=([01])",
            line,
        )
        if parsed is None:
            continue
        worker, mounts, jax_flax, ok = parsed.groups()
        by_worker[worker] = (int(mounts), int(jax_flax), int(ok))

    if not by_worker:
        thread_log("Environment check failed: no parseable env markers returned")
        return False

    failed = [worker for worker, (_, _, ok) in by_worker.items() if ok == 0]
    if failed:
        details = ", ".join(
            f"{worker}(mounts={mounts},jax_flax={jax_flax})"
            for worker, (mounts, jax_flax, _) in by_worker.items()
            if worker in failed
        )
        thread_log(f"Environment check failed on workers: {details}")
        return False

    thread_log("Environment check: Passed")
    return True


def check_datasets_mounts(tpu_name: str, zone: str, timeout: float = 15, max_ssh_tries: int=3) -> list[DatasetName] | None:
    """
    Check mount presence for multiple datasets.
    """
    script = read_remote_script("warmup.sh")
    cmd = f"ACTION=check_all bash -s <<'REMOTE_SCRIPT'\n{script}\nREMOTE_SCRIPT"

    try:
        result = gcloud_ssh(
            tpu_name, zone, cmd,
            worker="0", 
            timeout=timeout,
            capture_output=True,
            max_ssh_tries=max_ssh_tries,
        )
    except subprocess.TimeoutExpired:
        thread_log(f"SSH timed out: {tpu_name}")
        return
    if not result.ok:
        thread_log(f"SSH failed on {tpu_name}: {result}")
        return

    mounted_ds: list[DatasetName] = []
    for line in result.stdout.splitlines():
        parsed = re.fullmatch(
            r"(?:__TPURM_DATASET_STATUS__ )?(imagenet|fineweb)=([01])",
            line.strip(),
        )
        if parsed is None:
            continue
        dataset, value = parsed.groups()
        if value != "1":
            continue
        if dataset == "imagenet":
            mounted_ds.append("imagenet")
        else:
            mounted_ds.append("fineweb")

    return mounted_ds

def check_vacancy(tpu_name: str, zone: str, timeout: float = 15, max_ssh_tries: int=3) -> tuple[bool, str] | None:
    """
    SSH into worker 0 and check if TPU is in use.
    vacant=None means SSH failed.
    """
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
            worker="0", 
            timeout=timeout,
            capture_output=True,
            max_ssh_tries=max_ssh_tries,
        )
    except subprocess.TimeoutExpired:
        thread_log(f"SSH timed out: {tpu_name}")
        return
    if not result.ok:
        thread_log(f"SSH failed on {tpu_name}: {result}")
        return

    parts = result.stdout.split("---MARKER---")
    accel_output = parts[0].strip() if len(parts) > 0 else ""
    python_output = parts[1].strip() if len(parts) > 1 else ""
    load_output = parts[2].strip() if len(parts) > 2 else ""

    if python_output != "":
        python_examples = "\n".join(python_output.split("\n")[:4])
        thread_log(f"  {tpu_name}: python process examples:\n{python_examples}")

    load_1m = float(load_output.split()[0]) if load_output else 999.0
    load_5m = float(load_output.split()[1]) if load_output else 999.0
    vacant = (accel_output == "") and (python_output == "") and (load_5m < 2.0 or load_1m < 1.0)
    return vacant, load_output
