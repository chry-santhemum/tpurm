"""Lightweight TPU job scheduler."""

# Agents: Please DO NOT implement these functionalities unless told explicitly to do so.
# TODO: Feature list
# Build tests that are actually based on simulating the finite state machine
# If there are jobs queued, allocator should try to allocate according to the jobs queued, not randomly
# Call freeze when staging code / when to install dependencies
# Make common TPU ssh commands simple CLI, e.g. "tpu ssh --name ... --command ..."
# Delete Bucket folder after failure
# Relax restart constraint when job failed and no checkpoints saved
# Make allocator/schedule hparams part of the file state, hence modifiable mid-run


import copy
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from .common import (
    TPU, DatasetName, TPU_CONFIGS,
    _thread_local, thread_log, set_thread_vars,
    zone_to_region, name_to_tpu, size_to_family,
    gcloud_delete_tpu, gcloud_describe_tpu, list_tpus,
    check_datasets_mounts, check_env, check_vacancy,
)
from .state import FILE_STATE_DIR, FileState, Job, ManagedTPU
from .staging import (
    stage_code, launch, poll_launch, has_fatal_error_in_logs,
    kill_remote_processes, EXIT_CODE_FATAL, EXIT_CODE_SSH_RETRY,
)
from .steal import scan_target
from .initialize import allocate, ensure_ready
from .freeze import freeze



def tpu_matches_job(tpu: TPU, job: Job) -> bool:
    """Check if a TPU meets a job's size and region requirements."""
    tpu_region = zone_to_region(tpu.zone)
    if job.region and tpu_region not in job.region:
        return False
    tpu_family = size_to_family(tpu.size)
    tpu_chips = int(tpu.size.split('-')[1])
    for req_size in job.tpu_size:
        req_family = size_to_family(req_size)
        req_chips = int(req_size.split('-')[1])
        if tpu_family == req_family and tpu_chips >= req_chips and tpu_chips <= 2*req_chips:
            return True
    return False


def match_job(job: Job, tpus: dict[str, ManagedTPU], exclude: set[str] | None = None) -> str | None:
    """
    Job matching logic.
    
    Currently, we greedily find the next ManagedTPU with status free,
    sorted by ownership and then by the number of missing datasets.
    TODO: Find the overall best matching, instead of greedily.
    """
    if exclude is None:
        exclude = set()
    candidates: list[tuple[int, int, str]] = []
    for tpu_name, mt in tpus.items():
        if (
            mt.status != "free"
            or tpu_name in exclude
            or not tpu_matches_job(mt.tpu, job)
        ):
            continue

        # Count number of datasets missing
        missing_datasets = sum(1 for ds in job.datasets if ds not in mt.datasets)

        candidates.append((int(mt.owned), -missing_datasets, tpu_name))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][2]


def count_stolen_jobs(jobs: dict[int, Job], tpus: dict[str, ManagedTPU]) -> int:
    """Count the number of jobs matched to or running on stolen TPUs."""
    stolen_names = {tpu_name for tpu_name, mt in tpus.items() if not mt.owned}
    return sum(
        1 for j in jobs.values()
        if j.status in ("matched", "running")
        and j.assigned_tpu is not None
        and j.assigned_tpu.name in stolen_names
    )


def sync_state(file_state: FileState, startup: bool=False):
    # We first do a read, and only change the TPUs whose status remain the same
    read_status: dict[tuple[str, str], str] = {}  # key -> status
    # Keys for which to do all the checks
    check_all = []
    # Updated fields
    updates: dict[tuple[str, str], dict[str, Any]] = {}  # key -> update
    update_factory = {"status": None, "num_workers": None, "datasets": None}

    if startup:
        all_zones = set()
        for cfg in TPU_CONFIGS.values():
            all_zones.update(cfg["allowed_zones"])
        for zone in all_zones:
            for vm in list_tpus(zone):
                if vm.get("state", "") != "READY":
                    continue
                tpu_name = vm.get("name", "").rsplit("/", 1)[-1]
                tpu = name_to_tpu(tpu_name, zone)
                if tpu is None or tpu.owner != "atticusw":
                    continue
                key = (tpu.name, tpu.zone)
                updates[key] = copy.deepcopy(update_factory)
                check_all.append(key)
                thread_log(f"Discovered owned TPU: {tpu_name} ({zone})")

    _, jobs, tpus = file_state.snapshot()
    for name, mt in tpus.items():
        key = (name, mt.tpu.zone)
        read_status[key] = mt.status
        updates[key] = copy.deepcopy(update_factory)
        if startup:
            check_all.append(key)
        thread_log(f"Read status {key}: {mt.status}")
    # always mark TPUs with known running jobs as busy
    keys_with_running_jobs = []
    for job in jobs.values():
        if job.status == "running" and job.assigned_tpu is not None:
            keys_with_running_jobs.append((job.assigned_tpu.name, job.assigned_tpu.zone))
    
    if len(updates) == 0:
        return

    def sync_one(key: tuple[str, str]):
        tpu_name, zone = key
        out = updates[key]
        exists, state, health, num_workers = gcloud_describe_tpu(tpu_name, zone)
        if num_workers is not None:
            out["num_workers"] = num_workers
        if not exists:
            out["status"] = "untrack"
            return
        if (key in check_all) or (read_status[key] in ["need_init", "initializing"]):
            out["status"] = read_status.get(key, None)  # will be filled later
        if (key in check_all) or (read_status[key] not in ["need_init", "initializing"]):
            if state != "READY" or (health is not None and health != "HEALTHY"):
                out["status"] = "busy"
                return
            out["datasets"] = check_datasets_mounts(tpu_name, zone, max_ssh_tries=1)
            vacant_ok = check_vacancy(tpu_name, zone, max_ssh_tries=1)
            unknowns = int(out["datasets"] is None) + int(vacant_ok is None)
            if vacant_ok is None:
                out["status"] = "untrack" if unknowns >= 2 else "busy"
            elif not vacant_ok[0] or (key in keys_with_running_jobs):
                out["status"] = "busy"
            else:
                env_ok = check_env(tpu_name, zone, max_ssh_tries=1)
                unknowns += int(env_ok is None)
                if env_ok is None:
                    out["status"] = "untrack" if unknowns >= 2 else "busy"
                elif env_ok:
                    out["status"] = "free"
                else:
                    out["status"] = "need_init"
    
    parent_log_file = getattr(_thread_local, "log_file", None)
    def sync_one_with_context(key: tuple[str, str]) -> dict[str, Any]:
        with set_thread_vars(log_file=parent_log_file):
            return sync_one(key)
    
    with ThreadPoolExecutor(max_workers=min(32, len(updates))) as executor:
        futures = [executor.submit(sync_one_with_context, key) for key in updates.keys()]
        for fut in as_completed(futures):
            fut.result()

    with file_state.transact():
        for key, out in updates.items():
            if out["status"] is None:
                thread_log(f"Status is None for {key}, meaning sync_one failed for that TPU.")
                continue
            if key[0] not in file_state._tpus and out["status"] != "untrack":
                tpu = name_to_tpu(key[0], key[1])
                assert tpu is not None
                if out["num_workers"] is not None:
                    tpu.num_workers = out["num_workers"]
                file_state._tpus[key[0]] = ManagedTPU(
                    tpu=tpu,
                    owned=True,
                    status=out["status"],
                    datasets=out["datasets"] or [],
                )
                continue
            if out["status"] == "untrack":
                file_state._tpus.pop(key[0], None)
            elif file_state._tpus[key[0]].status == read_status[key]:
                # Only update when status has not been changed since read
                mt = file_state._tpus[key[0]]
                mt.status = out["status"]
                if out["num_workers"] is not None:
                    mt.tpu.num_workers = out["num_workers"]
                if out["datasets"] is not None:
                    mt.datasets = out["datasets"]


class Scheduler:
    def __init__(
        self,
        alloc_max: int,
        alloc_sizes: list[str],
        alloc_regions: list[str]|None,
        alloc_workers: int,
        init_workers: int,
        steal_wait: int,
        steal_max: int,
        tick_interval: int = 10,
        state_dir: Path = FILE_STATE_DIR,
    ):
        self.alloc_max = alloc_max
        self.alloc_sizes = alloc_sizes  # Default TPU sizes to allocate
        self.alloc_regions = alloc_regions  # Default regions to allocate in
        self.alloc_workers = alloc_workers
        self.init_workers = init_workers
        self.steal_wait = steal_wait
        self.steal_max = steal_max
        self.tick_interval = tick_interval
        self.state_dir = state_dir
        self.log_dir = state_dir / "logs"
        self.file_state = FileState(state_dir)

        # Termination signal
        self._stop_file = Path(self.state_dir) / "tpurm.stop"
        self._stop_event = threading.Event()

        # Communication with alloc workers
        # The workers can only pop from the queue
        # Event is set by the alloc monitor
        self._alloc_sleep_event = threading.Event()
        self._alloc_queue: list[tuple[str, str]] = []  # (size, zone)
        self._alloc_queue_lock = threading.Lock()

        # Communication with stealing
        self._steal_job: Job|None = None
        self._steal_target: tuple[TPU, float] | None = None  # (tpu, started_at)


    def startup(self):
        fs = self.file_state
        sync_state(fs, startup=True)

        with fs.transact():
            for tpu_name, mt in fs._tpus.items():
                if mt.status == "initializing":
                    thread_log(f"Resetting stuck initializing TPU {tpu_name} to need_init")
                    mt.status = "need_init"

        self.poll_jobs()
        self.drain_cancelled_jobs()
        sync_state(fs)


    def try_alloc(self, worker_id: int, size: str, zone: str) -> bool:
        tpu = allocate(size, zone, max_attempts=8, stop_events=[self._stop_event, self._alloc_sleep_event])
        if tpu is None:
            thread_log(f"[worker {worker_id}] Failed to allocate {size} in {zone}")
            return False
        thread_log(f"[worker {worker_id}] Allocated {tpu.name} in {zone}", force_print=True)
        with self.file_state.transact():
            self.file_state._tpus[tpu.name] = ManagedTPU(tpu=tpu, owned=True, status="need_init")
        return True

        
    def alloc_worker(self, worker_id: int):
        log_path = self.log_dir / f"alloc_worker_{worker_id}.log"
        default_combo = []
        for size in self.alloc_sizes:
            allowed_zones = TPU_CONFIGS[size_to_family(size)]["allowed_zones"]
            if self.alloc_regions is not None:
                allowed_zones = [z for z in allowed_zones if zone_to_region(z) in self.alloc_regions]
            default_combo.extend([(size, zone) for zone in allowed_zones])
        curr_pointer = 0

        with open(log_path, "w") as log_file, set_thread_vars(log_file=log_file):
            while True:
                if self._stop_event.is_set():
                    thread_log(f"[worker {worker_id}] Exiting.", force_print=True)
                    return
                if self._alloc_sleep_event.is_set():
                    self._stop_event.wait(30)
                    continue
                
                size, zone = None, None
                with self._alloc_queue_lock:
                    if self._alloc_queue:
                        size, zone = self._alloc_queue.pop(0)
                
                if size is not None:  # alloc queue was nonempty
                    assert zone is not None  # for type check
                    curr_pointer = 0
                else:
                    if len(default_combo) == 0:
                        self._stop_event.wait(30)
                        continue
                    size, zone = default_combo[curr_pointer]
                    curr_pointer = (curr_pointer + 1) % len(default_combo)
                
                self.try_alloc(worker_id, size, zone)


    def try_init(self, worker_id: int, mt: ManagedTPU) -> bool:
        tpu = mt.tpu
        skip_upgrade = not mt.owned

        if ensure_ready(tpu, skip_upgrade=skip_upgrade):
            thread_log(f"[worker {worker_id}] Successfully initialized: {tpu.name}", force_print=True)
            with self.file_state.transact():
                self.file_state._tpus[tpu.name] = ManagedTPU(tpu=tpu, owned=mt.owned, status="free")
            return True

        thread_log(f"[worker {worker_id}] Initialization failed. Deleting: {tpu.name}", force_print=True)
        gcloud_delete_tpu(tpu.name, tpu.zone)
        with self.file_state.transact():
            self.file_state._tpus.pop(tpu.name, None)
        return False


    def init_worker(self, worker_id: int):
        log_path = self.log_dir / f"init_worker_{worker_id}.log"
        with open(log_path, "a+") as log_file, set_thread_vars(log_file=log_file):
            while not self._stop_event.is_set():
                target = None
                with self.file_state.transact():
                    for _, mt in self.file_state._tpus.items():
                        if mt.status == "need_init":
                            mt.status="initializing"
                            target=mt
                            break
                if target is None:
                    self._stop_event.wait(30)
                    continue

                log_file.seek(0)
                log_file.truncate()
                log_file.flush()
                thread_log(f"[worker {worker_id}] Claimed: {target.tpu.name}", force_print=True)

                init_success = self.try_init(worker_id, target)
                thread_log(f"[worker {worker_id}] init success {init_success}: {target.tpu.name}", force_print=True)
                    
            thread_log(f"[worker {worker_id}] Exiting.", force_print=True)

    
    # Helpers

    def summary(self) -> tuple[str, int]:
        _, jobs, tpus = self.file_state.snapshot()
        n_queued_jobs = sum(1 for j in jobs.values() if j.status == "queued")
        n_owned_tpus = sum(1 for mt in tpus.values() if mt.owned)
        n_stolen_tpus = len(tpus) - n_owned_tpus
        header = f"{n_owned_tpus} owned, {n_stolen_tpus} stolen, {n_queued_jobs} queued jobs"

        # Build per-TPU status lines
        tpu_to_job = {}
        for j in jobs.values():
            if j.status == "running" and j.assigned_tpu:
                tpu_to_job[j.assigned_tpu.name] = j

        summary_lines = []
        for tpu_name, mt in tpus.items():
            job = tpu_to_job.get(tpu_name)
            if job:
                summary_lines.append(f"  {mt.tpu.name}: running job {job.job_id} ({job.run_name})")
            else:
                summary_lines.append(f"  {mt.tpu.name}: {mt.status}")
        
        summary = header
        if summary_lines:
            summary += "\n" + "\n".join(summary_lines)
        return summary, n_owned_tpus

    def drain_cancelled_jobs(self):
        _, jobs, tpus = self.file_state.snapshot()
        jobs_to_clear = []
        for job in jobs.values():
            if job.status != "cancelled" or job.assigned_tpu is None:
                continue

            tpu = job.assigned_tpu
            tracked_tpu = tpus.get(tpu.name)
            if tracked_tpu is None or tracked_tpu.status == "initializing":
                jobs_to_clear.append(job.job_id)
                continue

            thread_log(f"Killing processes for cancelled job {job.job_id} on {tpu.name}...")
            if kill_remote_processes(tpu.name, tpu.zone, job.log_dir):
                jobs_to_clear.append(job.job_id)
            else:
                thread_log(f"Cleanup failed for cancelled job {job.job_id} on {tpu.name}; will retry.")
        if len(jobs_to_clear) == 0:
            return

        # State transition: we only change the job status,
        # and leave the TPU status to be updated by sync_state.
        fs = self.file_state
        with fs.transact():
            for job_id in jobs_to_clear:
                job = fs._jobs.get(job_id)
                if job is None or job.status != "cancelled":
                    continue
                job.assigned_tpu = None


    def steal_tick(self):
        """Steal a TPU for the current _steal_job."""
        if self._steal_target is None:  # Look for fresh target
            job = self._steal_job  # TODO: deal with when job is cancelled here
            assert job is not None
            regions = []
            for tpu_size in job.tpu_size:
                family, _ = tpu_size.split("-")
                cfg = TPU_CONFIGS[family]
                regions.extend([zone_to_region(z) for z in cfg["allowed_zones"]])

            vacant_vms = scan_target(job.tpu_size, job.region or regions)
            vacant_tpus = [tpu for tpu in [name_to_tpu(name, zone) for name, zone in vacant_vms] if tpu is not None]
            _, _, tpus = self.file_state.snapshot()
            tracked_names = set(tpus.keys())
            vacant_tpus = [
                tpu for tpu in vacant_tpus
                if tpu.owner != "atticusw" and tpu.name not in tracked_names
            ]
            # Steal from the rich
            owner_freq = {}
            for tpu in vacant_tpus:
                owner_freq[tpu.owner] = owner_freq.get(tpu.owner, 0) + 1
            thread_log(f"Vacant TPU owners frequency: {owner_freq}")
            vacant_tpus.sort(key=lambda tpu: owner_freq[tpu.owner], reverse=True)

            if vacant_tpus:
                self._steal_target = (vacant_tpus[0], time.time())
                thread_log(f"Found vacant TPU {vacant_tpus[0].name}, will wait for {self.steal_wait}s before stealing.", force_print=True)

        else:
            job = self._steal_job
            assert job is not None
            if job.status != "queued":
                thread_log(f"Current steal target job {job.job_id} is no longer queued. Aborting steal.")
                self._steal_target = None
                return
            tpu, started_at = self._steal_target
            info = check_vacancy(tpu.name, tpu.zone)
            elapsed = time.time() - started_at
            if info is None or not info[0]:
                if info is None:
                    thread_log(f"TPU {tpu.name} died. Aborting steal.")
                else:
                    thread_log(f"TPU {tpu.name} became busy ({int(elapsed)}s / {self.steal_wait}s). Aborting steal.")
                self._steal_target = None
                with self.file_state.transact():
                    job = self.file_state._jobs[job.job_id]
                    job.assigned_tpu = None
                return
            if elapsed < self.steal_wait:
                thread_log(f"TPU {tpu.name} still vacant ({int(elapsed)}s / {self.steal_wait}s)")
                return
                
            # Wait complete; hand off to init workers
            with self.file_state.transact():
                job = self.file_state._jobs[job.job_id]
                if job.status == "queued":
                    thread_log(f"Stealing TPU {tpu.name}. Queued for init.", force_print=True)
                    job.assigned_tpu = tpu
                    job.status = "matched"
                    self.file_state._tpus[tpu.name] = ManagedTPU(tpu=tpu, owned=False, status="need_init")
            self._steal_target = None
            self._steal_job = None


    def launch_job(self, job_id: int, tpu_name: str):
        log_path = self.log_dir / f"job_{job_id}.log"
        with open(log_path, "a") as log_file, set_thread_vars(log_file=log_file):
            thread_log(f"[job {job_id}] Launching on {tpu_name}...", force_print=True)

            fs = self.file_state
            with fs.transact():
                tracked_job = fs._jobs[job_id]
                if (
                    tracked_job.status != "running"
                    or tracked_job.assigned_tpu is None
                    or tracked_job.assigned_tpu.name != tpu_name
                ):
                    thread_log(f"[job {job_id}] no longer runnable before launch, skipping launch")
                    return
                tracked_mt = fs._tpus.get(tpu_name)
                if tracked_mt is None:
                    thread_log(f"[job {job_id}] TPU {tpu_name} is no longer tracked before launch")
                    return
                job = copy.deepcopy(tracked_job)
                mt = copy.deepcopy(tracked_mt)

            if not ensure_ready(mt.tpu, skip_upgrade=not mt.owned):
                thread_log(f"[job {job_id}] readiness check failed before launch")
                self.finalize_job(job, EXIT_CODE_SSH_RETRY)
                return

            with fs.transact():
                tracked_job = fs._jobs[job_id]
                if (
                    tracked_job.status != "running"
                    or tracked_job.assigned_tpu is None
                    or tracked_job.assigned_tpu.name != tpu_name
                ):
                    thread_log(f"[job {job_id}] no longer runnable after readiness check, skipping launch")
                    return

            returncode = launch(
                mt.tpu, job.command, stage_dir=job.stage_dir,
                run_name=job.run_name, project_name=job.project_name,
                datasets=job.datasets,
            )
            if returncode != 0:  # Job failed to start
                self.finalize_job(job, returncode)
                return

    def finalize_job(self, job: Job, returncode: int):
        fs = self.file_state
        log_path = self.log_dir / f"job_{job.job_id}.log"

        with open(log_path, "a") as log_file, set_thread_vars(log_file=log_file):
            with fs.transact():
                tracked_job = fs._jobs[job.job_id]
                if tracked_job.status == "cancelled":
                    if tracked_job.assigned_tpu is not None:
                        mt = fs._tpus.get(tracked_job.assigned_tpu.name)
                        if mt is not None:
                            mt.status = "free"
                    tracked_job.assigned_tpu = None
                    thread_log(f"[job {tracked_job.job_id}] cancelled, skipping retry", force_print=True)
                    return

                tpu = tracked_job.assigned_tpu
                assert tpu is not None
                run_name = tracked_job.run_name
                project_name = tracked_job.project_name
                attempt = tracked_job.attempt
                max_att = tracked_job.max_att

            might_retry = (
                max_att > attempt
                and returncode != 0
                and returncode != EXIT_CODE_FATAL
            )
            new_stage_dir = None
            if might_retry:
                freeze()
                new_stage_dir = stage_code(run_name, project_name)

            tpu_dead = False
            if returncode == EXIT_CODE_SSH_RETRY:
                exists, state, health, _ = gcloud_describe_tpu(tpu.name, tpu.zone)
                if (not exists) or state != "READY" or (health is not None and health != "HEALTHY"):
                    thread_log(f"[job {job.job_id}] TPU {tpu.name} failed mid-task, removing it.", force_print=True)
                    tpu_dead = True

            with fs.transact():
                job = fs._jobs[job.job_id]
                if job.status == "cancelled":
                    if job.assigned_tpu is not None:
                        mt = fs._tpus.get(job.assigned_tpu.name)
                        if mt is not None:
                            mt.status = "free"
                    job.assigned_tpu = None
                    thread_log(f"[job {job.job_id}] cancelled, skipping retry", force_print=True)
                    return

                tpu = job.assigned_tpu
                assert tpu is not None
                mt = fs._tpus.get(tpu.name)
                if mt is not None:
                    mt.status = "free"
                    if tpu_dead:
                        fs._tpus.pop(mt.tpu.name, None)

                if returncode == 0:
                    job.status = "succeeded"
                    job.assigned_tpu = None
                    thread_log(f"[job {job.job_id}] succeeded", force_print=True)
                elif new_stage_dir is not None:
                    job.status = "queued"
                    job.assigned_tpu = None
                    job.attempt += 1
                    job.priority += 1
                    job.stage_dir = new_stage_dir
                    thread_log(f"[job {job.job_id}] failed and re-queued (exit {returncode})", force_print=True)
                else:
                    job.status = "failed"
                    job.assigned_tpu = None
                    thread_log(f"[job {job.job_id}] failed (exit {returncode})", force_print=True)


    def poll_jobs(self):
        _, jobs, _ = self.file_state.snapshot()
        completed: list[tuple[Job, int]] = []
        for job in jobs.values():
            if job.assigned_tpu is None or job.status != "running":
                continue
            if job.assigned_tpu.num_workers is None:
                thread_log(f"[job {job.job_id}] ERROR: TPU {job.assigned_tpu.name} has no num_workers.", force_print=True)
                continue

            returncode = poll_launch(job.log_dir, job.assigned_tpu.num_workers)
            if returncode is None and job.status == "running":
                exists, state, health, _ = gcloud_describe_tpu(job.assigned_tpu.name, job.assigned_tpu.zone)
                if (not exists) or state != "READY" or (health is not None and health != "HEALTHY"):
                    returncode = EXIT_CODE_SSH_RETRY
            if returncode is None:
                continue
            if returncode != 0 and has_fatal_error_in_logs(job.log_dir):
                returncode = EXIT_CODE_FATAL
            completed.append((job, returncode))

        for job, returncode in completed:
            self.finalize_job(job, returncode)


    def run_tick(self, prev_tick: str | None):
        if self._stop_file.exists():
            thread_log("Stop file detected, shutting down...", force_print=True)
            self._stop_event.set()
        if self._stop_event.is_set():
            return prev_tick

        self.poll_jobs()
        self.drain_cancelled_jobs()
        sync_state(self.file_state)

        tick, num_owned = self.summary()
        if tick != prev_tick:
            thread_log(tick, force_print=True)

        if num_owned >= self.alloc_max:
            self._alloc_sleep_event.set()
            thread_log(f"{num_owned} owned TPUs, alloc workers sleep")
        else:
            self._alloc_sleep_event.clear()
            thread_log(f"{num_owned} owned TPUs, alloc workers active")
        
        # Update job state
        fs = self.file_state
        launches: list[tuple[int, str]] = []
        with fs.transact():
            excluded_tpus = set()  # TPUs that already have matched jobs
            to_launch: list[tuple[int, str]] = []  # (job_id, tpu_name) to launch this turn

            # Check on jobs with "matched" status
            for job in fs._jobs.values():
                if job.status != "matched":
                    continue
                assert job.assigned_tpu is not None  # assigned_tpu is guaranteed to exist
                tpu_name = job.assigned_tpu.name
                if tpu_name not in fs._tpus:
                    job.assigned_tpu = None
                    job.status = "queued"
                else:
                    excluded_tpus.add(tpu_name)
                    if fs._tpus[tpu_name].status == "free":
                        to_launch.append((job.job_id, tpu_name))
            
            # Match queued jobs
            num_stolen_jobs = count_stolen_jobs(fs._jobs, fs._tpus)
            queued_jobs = [j for j in fs._jobs.values() if j.status == "queued"]
            queued_jobs.sort(key=lambda j: (-j.priority, j.created_at))

            first_unmatched: Job | None = None
            for job in queued_jobs:
                match = match_job(job, fs._tpus, exclude=excluded_tpus)
                # If matched a stolen TPU, check budget
                if match and not fs._tpus[match].owned:
                    if self.steal_max >= 0 and num_stolen_jobs >= self.steal_max:
                        match = None
                    else:
                        num_stolen_jobs += 1
                if match:
                    excluded_tpus.add(match)
                    to_launch.append((job.job_id, match))
                elif first_unmatched is None:
                    first_unmatched = job

            # Launch jobs and transition status
            for job_id, tpu_name in to_launch:
                job = fs._jobs[job_id]
                mt = fs._tpus[tpu_name]
                job.assigned_tpu = mt.tpu
                job.status = "running"
                mt.status = "busy"
                launches.append((job_id, tpu_name))

        for job_id, tpu_name in launches:
            self.launch_job(job_id, tpu_name)

        # Try stealing for first unmatched job if budget allows
        if (
            first_unmatched is not None
            and self.steal_wait >= 0
            and (self.steal_max < 0 or num_stolen_jobs < self.steal_max)
        ):
            if self._steal_job is None:
                self._steal_job = first_unmatched
            elif first_unmatched.job_id == self._steal_job.job_id:
                pass
            else:
                self._steal_job = first_unmatched
                self._steal_target = None
            self.steal_tick()

        self._stop_event.wait(self.tick_interval)
        
        return tick


    def stop(self):
        self._stop_event.set()
        self._stop_file.touch(exist_ok=True)


    def run(self):
        """Main daemon loop."""
        self._stop_file.unlink(missing_ok=True)  # remove previous stop file
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / "scheduler.log"

        with open(log_path, "a") as log_file, set_thread_vars(log_file=log_file):
            thread_log("Scheduler starting...", force_print=True)
            self.startup()

            thread_log(f"Starting {self.alloc_workers} alloc workers: sizes {self.alloc_sizes}, regions {self.alloc_regions}", force_print=True)
            self._threads = []
            # t = threading.Thread(target=self.monitor)
            # t.start()
            # self._threads.append(t)
            for i in range(self.alloc_workers):
                t = threading.Thread(target=self.alloc_worker, args=(i,))
                t.start()
                self._threads.append(t)
            for i in range(self.init_workers):
                t = threading.Thread(target=self.init_worker, args=(i,))
                t.start()
                self._threads.append(t)

            try:
                tick = None
                while not self._stop_event.is_set():
                    tick = self.run_tick(prev_tick=tick)

            except KeyboardInterrupt:
                thread_log("KeyboardInterrupt received. Shutting down scheduler.", force_print=True)
                self.stop()
            except Exception as e:
                thread_log(f"Scheduler crashed with unhandled exception: {e}", force_print=True)
                self.stop()
                raise
            finally:
                self.stop()
                for t in self._threads:
                    t.join(timeout=5)
