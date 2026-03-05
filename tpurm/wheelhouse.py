import hashlib
import shlex
from .common import (
    TPU, DEFAULT_KEYS_DIR, DEFAULT_SA_KEY_FILE,
    _thread_local,
    gcloud_ssh, thread_log, read_remote_script, run_cmd,
)

JAX_LINK = "https://storage.googleapis.com/jax-releases/libtpu_releases.html"


def _requirements_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def _env_string(tpu: TPU, requirements_lock: str, requirements_hash: str, wheelhouse_dir: str=""):
    """Build the env var prefix for remote commands."""
    gcs_prefix = f"{tpu.bucket}/atticusw/wheelhouse"
    sa_key_file = f"{DEFAULT_KEYS_DIR}/bucket-{tpu.region}.json"
    parts = [
        f"TAG={shlex.quote(tpu.wheelhouse_tag)}",
        f"GCS_PREFIX={shlex.quote(gcs_prefix)}",
        f"JAX_LINK={shlex.quote(JAX_LINK)}",
        f"REQUIREMENTS_LOCK={shlex.quote(requirements_lock)}",
        f"REQUIREMENTS_HASH={shlex.quote(requirements_hash)}",
        f"SERVICE_ACCOUNT={shlex.quote(tpu.service_account)}",
        f"SA_KEY_FILE={shlex.quote(sa_key_file)}",
        f"KEYS_DIR={shlex.quote(DEFAULT_KEYS_DIR)}",
        f"REGION={shlex.quote(tpu.region)}",
        f"DEFAULT_SA_KEY_FILE={shlex.quote(DEFAULT_SA_KEY_FILE)}",
    ]
    if wheelhouse_dir:
        parts.append(f"WHEELHOUSE_DIR={shlex.quote(wheelhouse_dir)}")
    return " ".join(parts)


def tarball_exists(tpu: TPU, requirements_hash: str) -> bool:
    uri = f"{tpu.bucket}/atticusw/wheelhouse/wheelhouse_{tpu.wheelhouse_tag}_{requirements_hash}.tar.gz"
    thread_log(f"Checking if tarball exists: {uri}")
    result = run_cmd(["gcloud", "storage", "ls", uri])
    if getattr(_thread_local, "dry_run", False):
        return True
    return result is not None and result.returncode == 0


def build(tpu: TPU, requirements_lock: str, wheelhouse_dir: str="") -> bool:
    """Build wheel tarball on worker 0 and upload to GCS."""
    req_hash = _requirements_hash(requirements_lock)
    thread_log(f"Requirements hash: {req_hash}")
    if tarball_exists(tpu, req_hash):
        thread_log(f"Tarball already exists for hash {req_hash}, skipping build")
        return True
    thread_log(f"Building tarball on {tpu.name} ({tpu.zone})")
    env = _env_string(tpu, requirements_lock, req_hash, wheelhouse_dir)
    preamble = read_remote_script("wheelhouse_preamble.sh")
    body = read_remote_script("wheelhouse_build.sh")
    full_cmd = f"{env} bash -s <<'REMOTE'\n" + preamble + '\n' + body + "\nREMOTE"
    result = gcloud_ssh(tpu.name, tpu.zone, full_cmd, worker="0")
    if getattr(_thread_local, "dry_run", False):
        return True
    return result is not None and result.returncode == 0


def install(tpu: TPU, requirements_lock: str, wheelhouse_dir: str="") -> bool:
    """Install wheels from GCS tarball on all workers."""
    thread_log(f"Installing tarball on {tpu.name} ({tpu.zone})")
    req_hash = _requirements_hash(requirements_lock)
    env = _env_string(tpu, requirements_lock, req_hash, wheelhouse_dir)
    preamble = read_remote_script("wheelhouse_preamble.sh")
    body = read_remote_script("wheelhouse_install.sh")
    full_cmd = f"{env} bash -s <<'REMOTE'\n" + preamble + '\n' + body + "\nREMOTE"
    result = gcloud_ssh(tpu.name, tpu.zone, full_cmd, worker="all")
    if getattr(_thread_local, "dry_run", False):
        return True
    return result is not None and result.returncode == 0
