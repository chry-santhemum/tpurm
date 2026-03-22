import hashlib
import shlex
from .common import (
    TPU, DEFAULT_KEYS_DIR,
    compose_remote_script, gcloud_ssh, thread_log, read_remote_script,
)

JAX_LINK = "https://storage.googleapis.com/jax-releases/libtpu_releases.html"


def requirements_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def _env_string(tpu: TPU, requirements_lock: str="", requirements_hash: str="", wheelhouse_dir: str=""):
    """Build the env var prefix for remote commands."""
    assert DEFAULT_KEYS_DIR, "DEFAULT_KEYS_DIR must be set"
    gcs_prefix = f"{tpu.bucket}/atticusw/wheelhouse"
    sa_key_file = f"{DEFAULT_KEYS_DIR}/bucket-{tpu.region}.json"
    parts = [
        f"TAG={shlex.quote(tpu.wheelhouse_tag)}",
        f"GCS_PREFIX={shlex.quote(gcs_prefix)}",
        f"JAX_LINK={shlex.quote(JAX_LINK)}",
        f"SERVICE_ACCOUNT={shlex.quote(tpu.service_account)}",
        f"SA_KEY_FILE={shlex.quote(sa_key_file)}",
        f"KEYS_DIR={shlex.quote(DEFAULT_KEYS_DIR)}",
        f"REGION={shlex.quote(tpu.region)}",
    ]
    if requirements_lock:
        parts.append(f"REQUIREMENTS_LOCK={shlex.quote(requirements_lock)}")
    if requirements_hash:
        parts.append(f"REQUIREMENTS_HASH={shlex.quote(requirements_hash)}")
    if wheelhouse_dir:
        parts.append(f"WHEELHOUSE_DIR={shlex.quote(wheelhouse_dir)}")
    return " ".join(parts)


def tarball_exists(tpu: TPU, requirements_hash: str) -> bool:
    uri = f"{tpu.bucket}/atticusw/wheelhouse/wheelhouse_{tpu.wheelhouse_tag}_{requirements_hash}.tar.gz"
    thread_log(f"Checking if tarball exists: {uri}")
    env = _env_string(tpu, requirements_hash=requirements_hash)
    preamble = compose_remote_script("gcloud_auth.sh", "wheelhouse_preamble.sh")
    full_cmd = f"{env} bash -s <<'REMOTE'\n{preamble}\ngcloud storage ls {shlex.quote(uri)} >/dev/null 2>&1\nREMOTE"
    result = gcloud_ssh(
        tpu.name, tpu.zone, full_cmd,
        operation="wheelhouse_exists",
        worker="0",
        timeout=60,
        capture_output=False,
        max_ssh_tries=3,
    )
    return result.ok


def build(tpu: TPU, requirements_lock: str, wheelhouse_dir: str="") -> bool:
    """Build wheel tarball on worker 0 and upload to GCS."""
    req_hash = requirements_hash(requirements_lock)
    thread_log(f"Requirements hash: {req_hash}")
    if tarball_exists(tpu, req_hash):
        thread_log(f"Tarball already exists for hash {req_hash}, skipping build")
        return True
    thread_log(f"Building tarball on {tpu.name} ({tpu.zone})")
    env = _env_string(tpu, requirements_lock=requirements_lock, requirements_hash=req_hash, wheelhouse_dir=wheelhouse_dir)
    preamble = compose_remote_script("gcloud_auth.sh", "wheelhouse_preamble.sh")
    body = read_remote_script("wheelhouse_build.sh")
    full_cmd = f"{env} bash -s <<'REMOTE'\n" + preamble + '\n' + body + "\nREMOTE"
    result = gcloud_ssh(
        tpu.name, tpu.zone, full_cmd,
        operation="wheelhouse_build",
        worker="0",
        timeout=None,
        capture_output=False,
        max_ssh_tries=3,
    )
    if result.retry_exhausted:
        thread_log(f"Wheelhouse build SSH retries exhausted on {tpu.name} (TPU likely preempted). Logs: {result.log_dir}")
    return result.ok


def install(tpu: TPU, requirements_lock: str, wheelhouse_dir: str="") -> bool:
    """Install wheels from GCS tarball on all workers."""
    thread_log(f"Installing tarball on {tpu.name} ({tpu.zone})")
    req_hash = requirements_hash(requirements_lock)
    env = _env_string(tpu, requirements_lock=requirements_lock, requirements_hash=req_hash, wheelhouse_dir=wheelhouse_dir)
    preamble = compose_remote_script("gcloud_auth.sh", "wheelhouse_preamble.sh")
    body = read_remote_script("wheelhouse_install.sh")
    full_cmd = f"{env} bash -s <<'REMOTE'\n" + preamble + '\n' + body + "\nREMOTE"
    result = gcloud_ssh(
        tpu.name, tpu.zone, full_cmd,
        operation="wheelhouse_install",
        worker="all",
        timeout=None,
        capture_output=False,
        max_ssh_tries=3,
    )
    if result.retry_exhausted:
        thread_log(f"Wheelhouse install SSH retries exhausted on {tpu.name} (TPU likely preempted). Logs: {result.log_dir}")
    return result.ok
