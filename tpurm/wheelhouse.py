import hashlib
import shlex

from .globals import DEFAULT_KEYS_DIR
from .tpu import TPU
from .util_log import LogContext
from .util_ssh import gcloud_ssh, read_remote_script

JAX_LINK = "https://storage.googleapis.com/jax-releases/libtpu_releases.html"


def requirements_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def _wheelhouse_env(
    tpu: TPU,
    *,
    requirements_lock: str = "",
    requirements_hash: str = "",
    wheelhouse_dir: str = "",
) -> dict[str, str]:
    assert DEFAULT_KEYS_DIR, "DEFAULT_KEYS_DIR must be set"
    gcs_prefix = f"{tpu.bucket}/atticusw/wheelhouse"
    sa_key_file = f"{DEFAULT_KEYS_DIR}/bucket-{tpu.region}.json"
    env = {
        "TAG": tpu.wheelhouse_tag,
        "GCS_PREFIX": gcs_prefix,
        "JAX_LINK": JAX_LINK,
        "SERVICE_ACCOUNT": tpu.service_account,
        "SA_KEY_FILE": sa_key_file,
        "KEYS_DIR": DEFAULT_KEYS_DIR,
        "REGION": tpu.region,
    }
    if requirements_lock:
        env["REQUIREMENTS_LOCK"] = requirements_lock
    if requirements_hash:
        env["REQUIREMENTS_HASH"] = requirements_hash
    if wheelhouse_dir:
        env["WHEELHOUSE_DIR"] = wheelhouse_dir
    return env


def _run_remote_scripts(
    tpu: TPU,
    *,
    script_names: list[str],
    env: dict[str, str],
    operation: str,
    worker: str,
    timeout: float | None,
    capture_output: bool = False,
    log_ctx: LogContext,
):
    env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    script = "\n".join(read_remote_script(name).rstrip() for name in script_names)
    remote_cmd = f"{env_prefix} bash -s <<'REMOTE_SCRIPT'\n{script}\nREMOTE_SCRIPT"
    return gcloud_ssh(
        tpu.name,
        tpu.zone,
        remote_cmd,
        operation=operation,
        worker=worker,
        timeout=timeout,
        capture_output=capture_output,
        max_ssh_tries=3,
        log_ctx=log_ctx,
    )


def tarball_exists(tpu: TPU, requirements_hash: str, *, log_ctx: LogContext) -> bool:
    uri = f"{tpu.bucket}/atticusw/wheelhouse/wheelhouse_{tpu.wheelhouse_tag}_{requirements_hash}.tar.gz"
    log_ctx.log(f"Checking if tarball exists: {uri}")
    result = _run_remote_scripts(
        tpu,
        script_names=["gcloud_auth.sh", "wheelhouse_preamble.sh", "wheelhouse_exists.sh"],
        env=_wheelhouse_env(tpu, requirements_hash=requirements_hash),
        operation="wheelhouse_exists",
        worker="0",
        timeout=60,
        capture_output=True,
        log_ctx=log_ctx,
    )
    if not result.ok:
        log_ctx.log(
            f"Wheelhouse existence check failed on {tpu.name}: "
            f"exit {result.returncode}. Logs: {result.log_dir}"
        )
        return False
    return "__TPURM_WHEELHOUSE_EXISTS__=1" in result.stdout


def build(tpu: TPU, requirements_lock: str, wheelhouse_dir: str = "", *, log_ctx: LogContext) -> bool:
    """Build wheel tarball on worker 0 and upload to GCS."""
    req_hash = requirements_hash(requirements_lock)
    log_ctx.log(f"Requirements hash: {req_hash}")
    if tarball_exists(tpu, req_hash, log_ctx=log_ctx):
        log_ctx.log(f"Tarball already exists for hash {req_hash}, skipping build")
        return True

    log_ctx.log(f"Building tarball on {tpu.name} ({tpu.zone})")
    result = _run_remote_scripts(
        tpu,
        script_names=["gcloud_auth.sh", "wheelhouse_preamble.sh", "wheelhouse_build.sh"],
        env=_wheelhouse_env(
            tpu,
            requirements_lock=requirements_lock,
            requirements_hash=req_hash,
            wheelhouse_dir=wheelhouse_dir,
        ),
        operation="wheelhouse_build",
        worker="0",
        timeout=None,
        log_ctx=log_ctx,
    )
    if result.ok:
        return True
    if result.retry_exhausted:
        log_ctx.log(f"Wheelhouse build SSH retries exhausted on {tpu.name}. Logs: {result.log_dir}")
    else:
        log_ctx.log(f"Wheelhouse build failed on {tpu.name}: exit {result.returncode}. Logs: {result.log_dir}")
    return False


def install(tpu: TPU, requirements_lock: str, wheelhouse_dir: str = "", *, log_ctx: LogContext) -> bool:
    """Install wheels from GCS tarball on all workers."""
    log_ctx.log(f"Installing tarball on {tpu.name} ({tpu.zone})")
    req_hash = requirements_hash(requirements_lock)
    result = _run_remote_scripts(
        tpu,
        script_names=["gcloud_auth.sh", "wheelhouse_preamble.sh", "wheelhouse_install.sh"],
        env=_wheelhouse_env(
            tpu,
            requirements_lock=requirements_lock,
            requirements_hash=req_hash,
            wheelhouse_dir=wheelhouse_dir,
        ),
        operation="wheelhouse_install",
        worker="all",
        timeout=None,
        log_ctx=log_ctx,
    )
    if result.ok:
        return True
    if result.retry_exhausted:
        log_ctx.log(f"Wheelhouse install SSH retries exhausted on {tpu.name}. Logs: {result.log_dir}")
    else:
        log_ctx.log(f"Wheelhouse install failed on {tpu.name}: exit {result.returncode}. Logs: {result.log_dir}")
    return False
