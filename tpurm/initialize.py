import shlex
import uuid
import time
import threading

from . import wheelhouse
from .globals import REPO_ROOT
from .tpu import TPU, AllocMode
from .util_gcloud import gcloud_create, gcloud_describe
from .util_log import LogContext
from .util_ssh import check_setup, gcloud_ssh, read_remote_script

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
_TERMINAL_TPU_STATES = {"PREEMPTED", "DELETING", "TERMINATED"}


def install_requirements(
    tpu: TPU,
    requirements_lock: str,
    requirements_hash: str,
    *,
    log_ctx: LogContext,
) -> bool:
    install_env = {
        "REQUIREMENTS_LOCK": requirements_lock,
        "REQUIREMENTS_HASH": requirements_hash,
    }
    install_success = False
    assert tpu.num_workers is not None, "TPU must be allocated before initialization."
    if tpu.wheelhouse_tag and tpu.num_workers > 1:
        if not wheelhouse.build(tpu, requirements_lock=requirements_lock, log_ctx=log_ctx):
            log_ctx.log("Warning: wheelhouse build failed")
        install_success = wheelhouse.install(
            tpu,
            requirements_lock=requirements_lock,
            log_ctx=log_ctx,
        )
        if not install_success:
            log_ctx.log("Warning: wheelhouse install failed; falling back to install.sh")
    if install_success:
        return True
    env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in install_env.items())
    script = read_remote_script("install.sh").rstrip()
    remote_cmd = f"{env_prefix} bash -s <<'REMOTE_SCRIPT'\n{script}\nREMOTE_SCRIPT"
    result = gcloud_ssh(
        tpu.name,
        tpu.zone,
        remote_cmd,
        operation="install",
        worker="all",
        timeout=None,
        max_ssh_tries=3,
        capture_output=False,
        log_ctx=log_ctx,
    )
    if not result.ok:
        log_ctx.log(f"install.sh failed on {tpu.name}: exit {result.returncode}. Logs: {result.log_dir}")
    return result.ok


def allocate(
    tpu_size: str, zone: str, max_attempts: int, 
    stop_events: list[threading.Event],
    log_ctx: LogContext,
    mode: AllocMode="spot", owner: str="atticusw",
) -> TPU|None:
    """
    Try to allocate a TPU with a given size and zone.

    Returns:
        tpu: The allocated TPU. The field `tpu.num_workers` is guaranteed to be not None.
    """
    mode_cfg = ALLOCATION_MODES[mode]
    attempt = 1
    while attempt <= max_attempts:
        for stop_event in stop_events:
            if stop_event.is_set():
                log_ctx.log("Stop event set, aborting allocation.")
                return None
        
        # Generate new ID each time
        tpu_id = str(uuid.uuid4()).replace("-", "")[:6]
        tpu = TPU(size=tpu_size, id=tpu_id, zone=zone, mode=mode, owner=owner)
        log_ctx.log(f"Trying to allocate TPU: {tpu.name} in {tpu.zone} (attempt {attempt}/{max_attempts})")

        result = gcloud_create(
            tpu.name,
            tpu.zone,
            accelerator_type=tpu.config["accelerator_type"](tpu.size),
            runtime_version=tpu.config["runtime_version"],
            service_account=tpu.service_account,
            mode_flag=mode_cfg["direct_flag"],
            log_ctx=log_ctx,
        )
        if result.returncode != 0:
            log_ctx.log(
                f"Error ({tpu.name}, {tpu.zone}): "
                f"Allocation failed (attempt {attempt}/{max_attempts}). Retrying in 5s."
            )
            attempt += 1
            continue

        # Get worker count
        time.sleep(15)
        info = gcloud_describe(tpu.name, tpu.zone, log_ctx=log_ctx)
        endpoints = info.get("networkEndpoints") if info is not None else None
        n_workers = len(endpoints) if isinstance(endpoints, list) else None
        if n_workers is None:
            log_ctx.log(
                f"Error ({tpu.name}, {tpu.zone}): "
                f"Allocation succeeded (attempt {attempt}/{max_attempts}), "
                f"but could not get worker count immediately after allocation. TPU likely preempted."
            )
            attempt += 1
            time.sleep(5)
            continue
        
        tpu.num_workers = n_workers
        log_ctx.log(f"Allocation successful ({tpu.name}, {tpu.zone}). Detected {n_workers} worker(s).")
        return tpu
    log_ctx.log(f"Error ({tpu.name}, {tpu.zone}): Failed to allocate TPU after {max_attempts} attempts.")


def init_and_install(
    tpu: TPU, *,
    requirements_lock: str, max_attempts: int, settle_time: int,
    skip_upgrade: bool = False,
    log_ctx: LogContext,
) -> bool:
    assert tpu.num_workers is not None, "TPU must be allocated before initialization."
    req_hash = wheelhouse.requirements_hash(requirements_lock)
    for attempt in range(1, max_attempts + 1):
        info = gcloud_describe(tpu.name, tpu.zone, log_ctx=log_ctx)
        state = info.get("state") if info is not None else None
        if info is None or state in _TERMINAL_TPU_STATES:
            log_ctx.log(f"TPU {tpu.name} does not exist, or is in terminal state {state}. Aborting initialization.")
            return False
        log_ctx.log(f"Initialization attempt {attempt}/{max_attempts}")

        setup = check_setup(tpu.name, tpu.zone, requirements_hash=req_hash, log_ctx=log_ctx)
        if setup is None:
            log_ctx.log("Warning: Setup check failed over SSH. Retrying...")
            continue
        if not setup["env"]:
            init_env = {"SKIP_UPGRADE": "1"} if skip_upgrade else {"SKIP_UPGRADE": "0"}
            env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in init_env.items())
            script = read_remote_script("init.sh").rstrip()
            remote_cmd = f"{env_prefix} bash -s <<'REMOTE_SCRIPT'\n{script}\nREMOTE_SCRIPT"
            result = gcloud_ssh(
                tpu.name,
                tpu.zone,
                remote_cmd,
                operation="init",
                worker="all",
                timeout=None,
                max_ssh_tries=3,
                capture_output=False,
                log_ctx=log_ctx,
            )
            if not result.ok:
                log_ctx.log(f"init.sh failed on {tpu.name}: exit {result.returncode}. Logs: {result.log_dir}")
                time.sleep(15)
                continue
            log_ctx.log(f"Waiting {settle_time}s for environment to settle...")
            time.sleep(settle_time)
            setup = check_setup(tpu.name, tpu.zone, requirements_hash=req_hash, log_ctx=log_ctx)
            if setup is None or not setup["env"]:
                log_ctx.log("Warning: Setup check failed after init. Retrying...")
                continue

        if not setup["requirements"]:
            if not install_requirements(tpu, requirements_lock, req_hash, log_ctx=log_ctx):
                time.sleep(30)
                continue
            setup = check_setup(tpu.name, tpu.zone, requirements_hash=req_hash, log_ctx=log_ctx)
            if setup is None or not setup["requirements"]:
                log_ctx.log("Warning: Setup check failed after install. Retrying...")
                continue
        log_ctx.log("Initialization successful.")
        return True

    log_ctx.log(f"Error: Failed to initialize all workers after {max_attempts} attempts")
    return False


def ensure_ready(tpu: TPU, skip_upgrade: bool = False, *, log_ctx: LogContext) -> bool:
    """Ensure TPU has correct environment. Returns True when ready."""
    requirements_lock = str(REPO_ROOT / "requirements.lock")
    req_hash = wheelhouse.requirements_hash(requirements_lock)
    setup = check_setup(tpu.name, tpu.zone, requirements_hash=req_hash, log_ctx=log_ctx)
    if setup is None or not setup["env"] or not setup["requirements"]:
        log_ctx.log(f"Initializing {tpu.name}...")
        if not init_and_install(
            tpu,
            requirements_lock=requirements_lock,
            max_attempts=5, settle_time=180,
            skip_upgrade=skip_upgrade,
            log_ctx=log_ctx,
        ):
            log_ctx.log(f"init_and_install failed for {tpu.name}")
            return False

    # Populate num_workers here
    if tpu.num_workers is None:
        info = gcloud_describe(tpu.name, tpu.zone, log_ctx=log_ctx)
        endpoints = info.get("networkEndpoints") if info is not None else None
        n_workers = len(endpoints) if isinstance(endpoints, list) else None
        if info is None or n_workers is None:
            log_ctx.log(f"Could not determine num_workers for ready TPU {tpu.name}")
            return False
        tpu.num_workers = n_workers
    return True
