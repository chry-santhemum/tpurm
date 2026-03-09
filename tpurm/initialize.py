"""VM initialization functions."""

import uuid
import time
import threading
import subprocess

from .common import (
    TPU, AllocMode, ALLOCATION_MODES, REPO_ROOT,
    thread_log, check_env,
    gcloud_ssh, gcloud_create_tpu, gcloud_describe_tpu,
    run_remote_script,
)
from . import wheelhouse

_TERMINAL_TPU_STATES = {"PREEMPTED", "DELETING", "TERMINATED"}


def _is_terminal_tpu_state(state: str | None) -> bool:
    return state in _TERMINAL_TPU_STATES


def allocate(
    tpu_size: str, zone: str, max_retries: int, 
    stop_events: list[threading.Event],
    mode: AllocMode="spot", owner: str="atticusw"
) -> TPU|None:
    """Try to allocate a TPU VM and return True if successful."""
    mode_cfg = ALLOCATION_MODES[mode]
    attempt = 1
    while attempt <= max_retries:
        for stop_event in stop_events:
            if stop_event.is_set():
                thread_log("Stop event set, aborting allocation.")
                return None

        tpu_id = str(uuid.uuid4()).replace("-", "")[:6]
        tpu = TPU(size=tpu_size, id=tpu_id, zone=zone, mode=mode, owner=owner)
        thread_log(f"Trying to allocate TPU: {tpu.name} in {tpu.zone} (attempt {attempt}/{max_retries})")

        result = gcloud_create_tpu(
            tpu.name, tpu.zone,
            tpu.config["accelerator_type"](tpu.size),
            tpu.config["runtime_version"],
            service_account=tpu.service_account,
            mode_flag=mode_cfg["direct_flag"],
        )
        if result is None or result.returncode != 0:
            thread_log(f"Allocation failed (attempt {attempt}/{max_retries}). Retrying in 10s.")
            attempt += 1
            if stop_event is None:
                time.sleep(10)
            else:
                stop_event.wait(10)
            continue

        thread_log(f"Allocation successful: {tpu.name} at {tpu.zone}")
        return tpu

    thread_log(f"Error: Failed to allocate TPU {tpu.name} after {max_retries} attempts.")



def wait_for_ssh(tpu_name: str, zone: str, poll_interval: int = 15, max_attempts: int = 20) -> bool:
    """Poll until SSH works on all workers. Returns True if successful."""
    for attempt in range(1, max_attempts + 1):
        result = gcloud_ssh(
            tpu_name, zone, "true",
            worker="all", timeout=30,
            capture_output=True,
            max_ssh_tries=1,
        )
        if result.ok:
            thread_log(f"SSH ready on {tpu_name} (attempt {attempt})")
            return True
        thread_log(f"SSH not ready on {tpu_name} (attempt {attempt}/{max_attempts}), retrying in {poll_interval}s...")
        time.sleep(poll_interval)
    thread_log(f"Error: SSH never became ready on {tpu_name}")
    return False


def reboot(tpu: TPU, boot_wait: int = 300) -> bool:
    """Issue 'sudo reboot' on all workers, wait, then poll until SSH works."""
    thread_log(f"Rebooting all workers on {tpu.name}...")
    try:
        result = gcloud_ssh(
            tpu.name, tpu.zone, "sudo reboot",
            worker="all",
            timeout=15,
            capture_output=False,
            ssh_flags=["-o ConnectionAttempts=1", "-o ConnectTimeout=5"],
            max_ssh_tries=1,
        )
    except subprocess.TimeoutExpired:
        exists, state, _ = gcloud_describe_tpu(tpu.name, tpu.zone)
        if not exists:
            thread_log(f"Reboot command timed out and TPU {tpu.name} no longer exists; skipping reboot wait.")
        elif _is_terminal_tpu_state(state):
            thread_log(f"Reboot command timed out and TPU {tpu.name} is in terminal state {state}; skipping reboot wait.")
        else:
            thread_log(f"Reboot command timed out for {tpu.name}; skipping reboot wait.")
        return False
    if not result.ok:
        if result.ssh_retry_exhausted:
            thread_log(f"Reboot SSH retries exhausted for {tpu.name}; TPU likely preempted.")
        exists, state, _ = gcloud_describe_tpu(tpu.name, tpu.zone)
        if not exists:
            thread_log(f"Reboot command failed and TPU {tpu.name} no longer exists; skipping reboot wait.")
        elif _is_terminal_tpu_state(state):
            thread_log(f"Reboot command failed and TPU {tpu.name} is in terminal state {state}; skipping reboot wait.")
        else:
            thread_log(f"Reboot command failed for {tpu.name}; skipping reboot wait.")
        return False

    thread_log(f"Restart issued. Sleeping for {boot_wait} seconds...")
    time.sleep(boot_wait)
    return wait_for_ssh(tpu.name, tpu.zone)


def init_and_install(
    tpu: TPU, *,
    requirements_lock: str, max_retries: int, settle_time: int,
    skip_upgrade: bool = False,
) -> bool:
    # Detect number of workers
    exists, _, n_workers = gcloud_describe_tpu(tpu.name, tpu.zone)
    if not exists:
        thread_log(f"Error: could not describe TPU {tpu.name}.")
        return False
    if n_workers is None:
        thread_log(f"Error: could not parse worker count for TPU {tpu.name}.")
        return False
    thread_log(f"Detected {n_workers} worker(s)")
    tpu.num_workers = n_workers

    for attempt in range(1, max_retries + 1):
        exists, state, _ = gcloud_describe_tpu(tpu.name, tpu.zone)
        if not exists:
            thread_log(f"TPU {tpu.name} no longer exists. Aborting initialization.")
            return False
        if _is_terminal_tpu_state(state):
            thread_log(f"TPU {tpu.name} entered terminal state {state}. Aborting initialization.")
            return False
        thread_log(f"Initialization attempt {attempt}/{max_retries}")
        
        # init
        init_env = {"SKIP_UPGRADE": "1"} if skip_upgrade else {"SKIP_UPGRADE": "0"}
        if not run_remote_script(tpu.name, tpu.zone, "init.sh", env=init_env, max_ssh_tries=3):
            thread_log("init.sh failed. Retrying in 15s.")
            time.sleep(15)
            continue
        
        # install
        install_success = False
        if tpu.wheelhouse_tag and n_workers > 1:
            if not wheelhouse.build(tpu, requirements_lock=requirements_lock):
                thread_log("Warning: wheelhouse build failed")
            install_success = wheelhouse.install(
                tpu,
                requirements_lock=requirements_lock,
            )
            if not install_success:
                thread_log("Warning: wheelhouse install failed; falling back to install.sh")
        if not install_success:
            if not run_remote_script(tpu.name, tpu.zone, "install.sh", env={"REQUIREMENTS_LOCK": requirements_lock}, max_ssh_tries=3):
                thread_log("install.sh failed. Retrying in 30s.")
                time.sleep(30)
                continue

        thread_log(f"Waiting {settle_time}s for environment to settle...")
        time.sleep(settle_time)
        if not check_env(tpu.name, tpu.zone):
            thread_log("Warning: Env check failed. Retrying...")
            continue
        thread_log("Initialization successful.")
        return True

    thread_log(f"Error: Failed to initialize all workers after {max_retries} attempts")
    return False


def ensure_ready(tpu: TPU, skip_upgrade: bool = False) -> bool:
    """Ensure TPU has correct environment. Returns True when ready."""
    if not check_env(tpu.name, tpu.zone):
        thread_log(f"Initializing {tpu.name}...")
        if not init_and_install(
            tpu,
            requirements_lock=str(REPO_ROOT / "requirements.lock"),
            max_retries=5, settle_time=180,
            skip_upgrade=skip_upgrade,
        ):
            thread_log(f"init_and_install failed for {tpu.name}")
            return False

    # Populate num_workers here
    if tpu.num_workers is None:
        exists, _, n_workers = gcloud_describe_tpu(tpu.name, tpu.zone)
        if not exists or n_workers is None:
            thread_log(f"Could not determine num_workers for ready TPU {tpu.name}")
            return False
        tpu.num_workers = n_workers
    return True
