import uuid
import time
import threading

from .common import (
    TPU, AllocMode, REPO_ROOT,
    thread_log, check_env, check_requirements,
    gcloud_create_tpu, gcloud_describe_tpu,
    run_remote_script,
)
from . import wheelhouse

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


def _install_requirements(tpu: TPU, requirements_lock: str, requirements_hash: str) -> bool:
    install_env = {
        "REQUIREMENTS_LOCK": requirements_lock,
        "REQUIREMENTS_HASH": requirements_hash,
    }
    install_success = False
    assert tpu.num_workers is not None, "TPU must be allocated before initialization."
    if tpu.wheelhouse_tag and tpu.num_workers > 1:
        if not wheelhouse.build(tpu, requirements_lock=requirements_lock):
            thread_log("Warning: wheelhouse build failed")
        install_success = wheelhouse.install(
            tpu,
            requirements_lock=requirements_lock,
        )
        if not install_success:
            thread_log("Warning: wheelhouse install failed; falling back to install.sh")
    if install_success:
        return True
    return run_remote_script(tpu.name, tpu.zone, "install.sh", env=install_env, max_ssh_tries=3)


def allocate(
    tpu_size: str, zone: str, max_attempts: int, 
    stop_events: list[threading.Event],
    mode: AllocMode="spot", owner: str="atticusw"
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
                thread_log("Stop event set, aborting allocation.")
                return None
        
        # Generate new ID each time
        tpu_id = str(uuid.uuid4()).replace("-", "")[:6]
        tpu = TPU(size=tpu_size, id=tpu_id, zone=zone, mode=mode, owner=owner)
        thread_log(f"Trying to allocate TPU: {tpu.name} in {tpu.zone} (attempt {attempt}/{max_attempts})")

        result = gcloud_create_tpu(
            tpu.name, tpu.zone,
            tpu.config["accelerator_type"](tpu.size),
            tpu.config["runtime_version"],
            service_account=tpu.service_account,
            mode_flag=mode_cfg["direct_flag"],
        )
        if result is None or result.returncode != 0:
            thread_log(
                f"Error ({tpu.name}, {tpu.zone}): "
                f"Allocation failed (attempt {attempt}/{max_attempts}). Retrying in 5s."
            )
            attempt += 1
            time.sleep(5)
            continue

        # Get worker count
        _, _, _, n_workers = gcloud_describe_tpu(tpu.name, tpu.zone)
        if n_workers is None:
            thread_log(
                f"Error ({tpu.name}, {tpu.zone}): "
                f"Allocation succeeded (attempt {attempt}/{max_attempts}), "
                f"but could not get worker count immediately after allocation. TPU likely preempted."
            )
            attempt += 1
            time.sleep(5)
            continue
        
        tpu.num_workers = n_workers
        thread_log(f"Allocation successful ({tpu.name}, {tpu.zone}). Detected {n_workers} worker(s).")
        return tpu
    thread_log(f"Error ({tpu.name}, {tpu.zone}): Failed to allocate TPU after {max_attempts} attempts.")


def init_and_install(
    tpu: TPU, *,
    requirements_lock: str, max_attempts: int, settle_time: int,
    skip_upgrade: bool = False,
) -> bool:
    assert tpu.num_workers is not None, "TPU must be allocated before initialization."
    req_hash = wheelhouse.requirements_hash(requirements_lock)
    for attempt in range(1, max_attempts + 1):
        exists, state, _, _ = gcloud_describe_tpu(tpu.name, tpu.zone)
        if not exists or state in _TERMINAL_TPU_STATES:
            thread_log(f"TPU {tpu.name} does not exist, or is in terminal state {state}. Aborting initialization.")
            return False
        thread_log(f"Initialization attempt {attempt}/{max_attempts}")

        base_env_ok = check_env(tpu.name, tpu.zone)
        if base_env_ok is None:
            thread_log("Warning: Base environment check failed over SSH. Retrying...")
            continue
        if not base_env_ok:
            init_env = {"SKIP_UPGRADE": "1"} if skip_upgrade else {"SKIP_UPGRADE": "0"}
            if not run_remote_script(tpu.name, tpu.zone, "init.sh", env=init_env, max_ssh_tries=3):
                thread_log("init.sh failed. Retrying in 15s.")
                time.sleep(15)
                continue
            thread_log(f"Waiting {settle_time}s for environment to settle...")
            time.sleep(settle_time)
            base_env_ok = check_env(tpu.name, tpu.zone)
            if base_env_ok is not True:
                thread_log("Warning: Base environment check failed after init. Retrying...")
                continue

        requirements_ok = check_requirements(tpu.name, tpu.zone, req_hash)
        if requirements_ok is None:
            thread_log("Warning: Requirements check failed over SSH. Retrying...")
            continue
        if not requirements_ok:
            if not _install_requirements(tpu, requirements_lock, req_hash):
                thread_log("install.sh failed. Retrying in 30s.")
                time.sleep(30)
                continue
            if not check_requirements(tpu.name, tpu.zone, req_hash):
                thread_log("Warning: Requirements check failed after install. Retrying...")
                continue
        thread_log("Initialization successful.")
        return True

    thread_log(f"Error: Failed to initialize all workers after {max_attempts} attempts")
    return False


def ensure_ready(tpu: TPU, skip_upgrade: bool = False) -> bool:
    """Ensure TPU has correct environment. Returns True when ready."""
    requirements_lock = str(REPO_ROOT / "requirements.lock")
    req_hash = wheelhouse.requirements_hash(requirements_lock)
    if check_env(tpu.name, tpu.zone) is not True or check_requirements(tpu.name, tpu.zone, req_hash) is not True:
        thread_log(f"Initializing {tpu.name}...")
        if not init_and_install(
            tpu,
            requirements_lock=requirements_lock,
            max_attempts=5, settle_time=180,
            skip_upgrade=skip_upgrade,
        ):
            thread_log(f"init_and_install failed for {tpu.name}")
            return False

    # Populate num_workers here
    if tpu.num_workers is None:
        exists, _, _, n_workers = gcloud_describe_tpu(tpu.name, tpu.zone)
        if not exists or n_workers is None:
            thread_log(f"Could not determine num_workers for ready TPU {tpu.name}")
            return False
        tpu.num_workers = n_workers
    return True
