"""Lightweight TPU job scheduler."""

# Agents: Please DO NOT implement these functionalities unless told explicitly to do so.
# TODO: Feature list
# Match jobs down the list as well, not eagerly
# Call freeze when staging code / when to install dependencies
# Make common TPU ssh commands simple CLI, e.g. "tpu ssh --name ... --command ..."
# Delete Bucket folder after failure
# Relax restart constraint when job failed and no checkpoints saved
# Make allocator/schedule hparams part of the file state, hence modifiable mid-run
# If there are jobs queued, allocator should try to allocate according to the jobs queued, not randomly

import contextlib
import copy
import fcntl
import json
import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Any, Literal

from .common import (
    TPU, REPO_ROOT, TPU_CONFIGS,
    thread_log, set_thread_vars, zone_to_region, name_to_tpu, size_to_family,
    gcloud_delete_tpu, gcloud_describe_tpu, list_tpus, check_data_mount, check_vacancy,
)
from .staging import stage_code, launch, kill_remote_processes, EXIT_CODE_FATAL, EXIT_CODE_SSH_RETRY
from .steal import scan_target
from .initialize import allocate, ensure_ready, reboot, check_env

FILE_STATE_DIR = str(REPO_ROOT / ".tpurm")
JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled", "orphaned"]

@dataclass
class ManagedTPU:
    tpu: TPU
    owned: bool
    status: str   # need_init | initializing | free | busy

@dataclass
class Job:
    # Properties known at creation time (should not be changed)
    job_id: int
    created_at: float
    command: str
    stage_dir: str
    tpu_size: list[str]
    region: list[str] | None
    run_name: str
    project_name: str
    max_retry: int  # 0 = no retry
    # Mutable properties
    status: JobStatus
    assigned_tpu: TPU | None  # last assigned
    attempt: int
    priority: int  # higher is more prioritized

def _deserialize(d: Any):
    # convenience for loading file state
    if isinstance(d, dict) and all(k in d for k in ["size", "mode", "owner", "id", "zone"]):
        tpu = TPU(d["size"], d["mode"], d["owner"], d["id"], d["zone"])
        if d.get("name") and d["name"] != tpu.name:
            tpu.name = d["name"]
        return tpu
    else:
        return d

class FileState:
    next_job_id: int
    jobs: dict[int, Job]  # job_id -> Job
    tpus: dict[str, ManagedTPU]  # tpu_name -> ManagedTPU

    def __init__(self, state_dir: str):
        self.state_dir = state_dir
        Path(state_dir).mkdir(parents=True, exist_ok=True)

        self._path = Path(state_dir) / "state.json"
        self._lock_path = Path(state_dir) / "state.lock"
        self._mutex = threading.RLock()
        self.read()
    
    # Public, thread-safe methods
    def read(self):
        """Creates a new copy in-memory every time."""
        with self._mutex:
            with open(self._lock_path, "a") as lf:
                fcntl.flock(lf, fcntl.LOCK_SH)
                self._load_unlocked()

    @contextlib.contextmanager
    def transact(self):
        """Read and write."""
        with self._mutex:
            with open(self._lock_path, "w") as lf:
                fcntl.flock(lf, fcntl.LOCK_EX)
                self._load_unlocked()
                yield self
                self._save_unlocked()

    def snapshot(self) -> tuple[dict[int, Job], dict[str, ManagedTPU]]:
        """Return deep-copied jobs/tpus from a consistent shared-lock read."""
        with self._mutex:
            with open(self._lock_path, "a") as lf:
                fcntl.flock(lf, fcntl.LOCK_SH)
                self._load_unlocked()
                return copy.deepcopy(self.jobs), copy.deepcopy(self.tpus)
    
    # These methods are NOT thread-safe
    def _load_unlocked(self):
        if not self._path.exists():
            self.next_job_id = 1
            self.jobs = {}
            self.tpus = {}
            return
        with open(self._path) as f:
            data = json.load(f)
        self.next_job_id = data["next_job_id"]
        self.jobs = {int(jid): Job(**{k: _deserialize(v) for k, v in job.items()}) for jid, job in data["jobs"].items()}
        self.tpus = {tpu_name: ManagedTPU(**{k: _deserialize(v) for k, v in tpu.items()}) for tpu_name, tpu in data["tpus"].items()}

    def _save_unlocked(self):
        data = {
            "next_job_id": self.next_job_id,
            "jobs": {str(jid): asdict(job) for jid, job in self.jobs.items()},
            "tpus": {tpu_name: asdict(tpu) for tpu_name, tpu in self.tpus.items()},
        }
        tmp = str(self._path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self._path)


def sync_tracked_tpus(file_state: FileState, startup: bool=False):
    """
    Scan and sync TPU status.
    Only modifies the file_state.tpus dict.

    If startup: scan all zones to check for TPUs with my name.
    """
    update_status: dict[tuple[str, str], str] = {}  # (tpu_name, zone) -> status
    read_status: dict[tuple[str, str], str] = {}
    owned: list[TPU] = []

    # Scan all the zones to find owned TPUs
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
                thread_log(f"Discovered owned TPU: {tpu_name} ({zone})")
                owned.append(tpu)

    jobs_snapshot, tpus_snapshot = file_state.snapshot()
    running_tpu_names = {
        j.assigned_tpu.name for j in jobs_snapshot.values()
        if j.status == "running" and j.assigned_tpu
    }
    for tpu_name, mt in tpus_snapshot.items():
        read_status[(tpu_name, mt.tpu.zone)] = mt.status
    for tpu_name, mt in tpus_snapshot.items():
        key = (tpu_name, mt.tpu.zone)
        if mt.status in ["initializing", "need_init"]:
            if not gcloud_describe_tpu(*key)[0]:
                thread_log(f"{mt.tpu.name} was preempted during initialization. Untracking.")
                update_status[key] = "untrack"
            continue  # Don't change the status otherwise
        if mt.owned and (mt.tpu.name not in [t.name for t in owned]):
            owned.append(mt.tpu)
        # Stolen TPUs
        elif (check_data_mount(*key) and check_env(*key)):
            if check_vacancy(*key)["vacant"]:
                thread_log(f"Previously-stolen TPU {mt.tpu.name} seems to be free.")
                update_status[key] = "free"
            else:
                thread_log(f"Previously-stolen TPU {mt.tpu.name} has a mount but is busy.")
                update_status[key] = "busy"
        else:
            thread_log(f"Previously-stolen TPU {mt.tpu.name} doesn't have the env anymore. Untracking it.")
            update_status[key] = "untrack"

    # Owned TPUs
    for tpu in owned:
        key = (tpu.name, tpu.zone)
        if not gcloud_describe_tpu(*key)[0]:
            thread_log(f"Owned TPU {tpu.name} ({tpu.zone}) no longer exists.")
            update_status[key] = "untrack"
            continue
        if not check_vacancy(*key)["vacant"]:
            update_status[key] = "busy"
        elif check_data_mount(*key) and check_env(*key):
            update_status[key] = "busy" if tpu.name in running_tpu_names else "free"
        else:
            update_status[key] = "need_init"

    # Write the file state
    with file_state.transact():
        to_untrack = set()
        for tpu_name, mt in file_state.tpus.items():
            key = (tpu_name, mt.tpu.zone)
            if mt.status != read_status.get(key) or key not in update_status:
                continue
            elif update_status[key] == "untrack":
                to_untrack.add(tpu_name)
            else:
                mt.status = update_status[key]
        for tpu_name in to_untrack:
            del file_state.tpus[tpu_name]

        # Add TPUs not in the file state
        tracked_names = list(file_state.tpus.keys())
        for (name, zone), status in update_status.items():
            if name not in tracked_names and status != "untrack":
                tpu = name_to_tpu(name, zone)
                if tpu is None:
                    continue
                file_state.tpus[tpu.name] = ManagedTPU(tpu=tpu, owned=True, status=status)


def submit_job(
    tpu_size: list[str], region: list[str]|None,
    run_name: str, project_name: str,
    command: Optional[str], command_path: Optional[str],
    priority: int = 0, max_retry: int = 0,
    state_dir: str = FILE_STATE_DIR,
) -> int:
    assert (command_path is None) ^ (command is None), "Exactly one of command and command_path must be provided"
    if command is None:
        command = Path(command_path).read_text().strip()  # type: ignore
    if command.startswith("python "):
        raise ValueError("Use python3.13 instead of python.")
    stage_dir = stage_code(run_name, project_name)

    fs = FileState(state_dir)
    with fs.transact():
        job_id = fs.next_job_id
        fs.next_job_id += 1
        fs.jobs[job_id] = Job(
            job_id=job_id,
            created_at=time.time(),
            command=command,
            stage_dir=stage_dir,
            tpu_size=tpu_size,
            region=region,
            run_name=run_name,
            project_name=project_name,
            max_retry=max_retry,
            priority=priority,
            status="queued",
            assigned_tpu=None,
            attempt=0,
        )
    thread_log(f"Submitted job {job_id}: run_name={run_name}")
    return job_id


def kill_local_processes(tpu_name: str):
    """Kill local processes that reference tpu_name."""
    result = subprocess.run(["pgrep", "-af", tpu_name], capture_output=True, text=True)
    pattern = re.compile(rf"{re.escape(tpu_name)}([^0-9]|$)")
    for line in result.stdout.strip().splitlines():
        parts = line.split(None, 1)
        pid_str, cmdline = parts[0], parts[1] if len(parts) > 1 else ""
        if not pattern.search(cmdline):
            continue
        try:
            pid = int(pid_str)
            if pid != os.getpid():
                os.kill(pid, signal.SIGKILL)
                thread_log(f"Killed local process {pid}: {cmdline[:80]}")
        except (ValueError, ProcessLookupError):
            pass

def cancel_job(job_id: int, state_dir: str = FILE_STATE_DIR):
    fs = FileState(state_dir)

    tpu_to_kill = None
    with fs.transact():
        if job_id not in fs.jobs:
            thread_log(f"Job {job_id} not found.")
            return
        job = fs.jobs[job_id]
        if job.status == "queued":
            job.status = "cancelled"
            thread_log(f"Cancelled queued job {job_id}")
            return
        elif job.status == "running":
            tpu_to_kill = job.assigned_tpu
            job.status = "cancelled"
        else:
            thread_log(f"Job {job_id} is already {job.status}, nothing to cancel.")
            return

    if tpu_to_kill:
        thread_log(f"Killing processes for job {job_id} on {tpu_to_kill.name}...")
        kill_local_processes(tpu_to_kill.name)
        kill_remote_processes(tpu_to_kill.name, tpu_to_kill.zone)
        thread_log(f"Cancelled running job {job_id}")


class Allocator:
    """Pre-allocates TPUs regardless of queued jobs."""

    def __init__(
        self,
        target_num: int,
        sizes: list[str],
        regions: list[str]|None,
        max_alloc_workers: int,
        max_init_workers: int,
        tick_interval: int,
        state_dir: str=FILE_STATE_DIR
    ):
        self.target_num = target_num
        self.sizes = sizes
        self.regions = regions
        self.max_alloc_workers = max_alloc_workers
        self.max_init_workers = max_init_workers
        self.tick_interval = tick_interval
        self.state_dir = state_dir
        self.log_dir = Path(state_dir) / "logs"
        self.file_state = FileState(state_dir=state_dir)
        self._stop_file = Path(self.state_dir) / "tpurm.stop"
        self._stop_event = threading.Event()
        self._unpause_event = threading.Event()

    def try_init(self, worker_id: int, mt: ManagedTPU) -> bool:
        tpu = mt.tpu
        skip_upgrade = not mt.owned

        if ensure_ready(tpu, skip_upgrade=skip_upgrade):
            thread_log(f"[worker {worker_id}] Successfully initialized: {tpu.name}", force_print=True)
            with self.file_state.transact():
                tracked = self.file_state.tpus.get(tpu.name)
                if tracked is not None:
                    tracked.status = "free"
            return True

        thread_log(f"[worker {worker_id}] Initialization failed, rebooting: {tpu.name}", force_print=True)
        if not reboot(tpu):
            thread_log(f"[worker {worker_id}] Reboot failed, deleting: {tpu.name}", force_print=True)
            gcloud_delete_tpu(tpu.name, tpu.zone)
            with self.file_state.transact():
                self.file_state.tpus.pop(tpu.name, None)
            return False
        
        # Retry after reboot
        if ensure_ready(tpu, skip_upgrade=skip_upgrade):
            thread_log(f"[worker {worker_id}] Successfully initialized after reboot: {tpu.name}", force_print=True)
            with self.file_state.transact():
                tracked = self.file_state.tpus.get(tpu.name)
                if tracked is not None:
                    tracked.status = "free"
            return True

        thread_log(f"[worker {worker_id}] Initialization still failing after reboot, deleting: {tpu.name}", force_print=True)
        gcloud_delete_tpu(tpu.name, tpu.zone)
        with self.file_state.transact():
            self.file_state.tpus.pop(tpu.name, None)
        return False

    def init_worker(self, worker_id: int):
        log_path = self.log_dir / f"init_worker_{worker_id}.log"
        last_claimed_tpu_name: str | None = None
        with open(log_path, "a+") as log_file, set_thread_vars(log_file=log_file):
            while not self._stop_event.is_set():
                target = None
                with self.file_state.transact():
                    for tpu_name, mt in self.file_state.tpus.items():
                        if mt.status == "need_init":
                            mt.status="initializing"
                            target=mt
                            break
                if target is None:
                    self._stop_event.wait(self.tick_interval)
                    continue

                if target.tpu.name != last_claimed_tpu_name:
                    log_file.seek(0)
                    log_file.truncate()
                    log_file.flush()
                    last_claimed_tpu_name = target.tpu.name
                thread_log(f"[worker {worker_id}] Claimed: {target.tpu.name}", force_print=True)
                try:
                    init_success = self.try_init(worker_id, target)
                    thread_log(f"[worker {worker_id}] init success {init_success}: {target.tpu.name}", force_print=True)
                except Exception as e:
                    thread_log(f"[worker {worker_id}] Error during initialization: {target.tpu.name} {e}", force_print=True)
                    exists, state = gcloud_describe_tpu(target.tpu.name, target.tpu.zone)
                    should_untrack = (not exists) or (state in {"PREEMPTED", "DELETING", "TERMINATED"})
                    with self.file_state.transact():
                        if should_untrack:
                            self.file_state.tpus.pop(target.tpu.name, None)
                        else:
                            tracked = self.file_state.tpus.get(target.tpu.name)
                            if tracked is not None:
                                tracked.status = "need_init"
                    
            thread_log(f"[worker {worker_id}] Exiting.", force_print=True)

    def alloc_worker(self, worker_id: int):
        log_path = self.log_dir / f"alloc_worker_{worker_id}.log"
        with open(log_path, "w") as log_file, set_thread_vars(log_file=log_file):
            while True:
                for size in self.sizes:
                    allowed_zones = TPU_CONFIGS[size_to_family(size)]["allowed_zones"]
                    if self.regions is not None:
                        allowed_zones = [z for z in allowed_zones if zone_to_region(z) in self.regions]
                    for zone in allowed_zones:
                        while True:
                            if self._stop_event.is_set():
                                thread_log(f"[worker {worker_id}] Exiting.", force_print=True)
                                return

                            tpu = allocate(size, zone, max_retries=8, stop_event=self._stop_event, unpause_event=self._unpause_event)
                            if tpu is None:
                                break

                            thread_log(f"[worker {worker_id}] Allocated {tpu.name} in {zone}. Queued for init.", force_print=True)
                            with self.file_state.transact():
                                self.file_state.tpus[tpu.name] = ManagedTPU(tpu=tpu, owned=True, status="need_init")

    def monitor(self):
        log_path = self.log_dir / "alloc_monitor.log"
        startup = True
        with open(log_path, "a") as log_file, set_thread_vars(log_file=log_file):
            while not self._stop_event.is_set():
                sync_tracked_tpus(self.file_state, startup=startup)
                if startup:
                    with self.file_state.transact():
                        for tpu_name, mt in self.file_state.tpus.items():
                            if mt.status == "initializing":
                                thread_log(f"Resetting stuck initializing TPU {tpu_name} to need_init")
                                mt.status = "need_init"
                self.file_state.read()
                num_owned = sum(1 for mt in self.file_state.tpus.values() if mt.owned)
                if num_owned >= self.target_num:
                    self._unpause_event.clear()
                    thread_log(f"{num_owned}/{self.target_num} owned TPUs alive, pausing alloc workers")
                else:
                    self._unpause_event.set()
                    thread_log(f"{num_owned}/{self.target_num} owned TPUs alive, alloc workers active")
                startup = False
                self._stop_event.wait(self.tick_interval)
                if self._stop_file.exists():
                    thread_log("Stop file detected, shutting down allocator...")
                    self._stop_event.set()

    def run(self):
        """Launch monitor, alloc worker, and init worker threads. Returns immediately."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._unpause_event.set()  # start awake

        thread_log(
            f"Allocator starting: target={self.target_num}, sizes={self.sizes}, "
            f"alloc_workers={self.max_alloc_workers}, init_workers={self.max_init_workers}",
            force_print=True,
        )
        self._threads = []
        t = threading.Thread(target=self.monitor)
        t.start()
        self._threads.append(t)
        for i in range(self.max_alloc_workers):
            t = threading.Thread(target=self.alloc_worker, args=(i,))
            t.start()
            self._threads.append(t)
        for i in range(self.max_init_workers):
            t = threading.Thread(target=self.init_worker, args=(i,))
            t.start()
            self._threads.append(t)

    def join(self):
        for t in self._threads:
            t.join()

    def stop(self):
        self._stop_event.set()
        self._unpause_event.set()


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

class Scheduler:
    def __init__(
        self,
        steal_wait: int,
        max_steal: int,
        tick_interval: int,
        state_dir: str = FILE_STATE_DIR,
    ):
        self.steal_wait = steal_wait
        self.max_steal = max_steal
        self.tick_interval = tick_interval
        self.state_dir = state_dir
        self.log_dir = Path(state_dir) / "logs"
        self.file_state = FileState(state_dir=state_dir)
        self._job_threads: dict[int, threading.Thread] = {}  # job threads that are alive
        self._stop_file = Path(self.state_dir) / "tpurm.stop"
        self._stop_event = threading.Event()
        self._steal_target: tuple[TPU, float] | None = None  # (tpu, started_at)

    def startup(self):
        fs = self.file_state
        sync_tracked_tpus(fs, startup=True)

        with fs.transact():
            for jid, job in list(fs.jobs.items()):
                if job.status != "running":
                    continue
                tpu_name = job.assigned_tpu.name
                thread_log(f"Job {jid} was running on {tpu_name} before daemon was shut down. Marking orphaned.")
                job.status = "orphaned"

    def match_tpu(
        self,
        job: Job,
        tpu_map: dict[str, ManagedTPU],
        exclude: set[str] | None = None
    ) -> ManagedTPU | None:
        """Match a job to a free TPU, preferring owned over stolen."""
        if exclude is None:
            exclude = set()
        # Owned
        for tpu_name, mt in tpu_map.items():
            if mt.status != "free" or not mt.owned or tpu_name in exclude:
                continue
            if tpu_matches_job(mt.tpu, job):
                return mt
        # Stolen
        for tpu_name, mt in tpu_map.items():
            if mt.status != "free" or mt.owned or tpu_name in exclude:
                continue
            if tpu_matches_job(mt.tpu, job):
                return mt
        return None

    def reap_threads(self):
        finished = []
        for jid, t in self._job_threads.items():
            if not t.is_alive():
                t.join(timeout=1)
                finished.append(jid)
        for jid in finished:
            del self._job_threads[jid]
            thread_log(f"Reaped thread for job {jid}", force_print=True)

    def steal_tick(self, job: Job):
        """Scan for vacant TPUs to steal. After wait period, mark as need_init."""
        if self._steal_target is None:  # Look for fresh target
            regions = []
            for tpu_size in job.tpu_size:
                family, _ = tpu_size.split("-")
                cfg = TPU_CONFIGS[family]
                regions.extend([zone_to_region(z) for z in cfg["allowed_zones"]])

            vacant_vms = scan_target(job.tpu_size, job.region or regions)
            vacant_tpus = [tpu for tpu in [name_to_tpu(name, zone) for name, zone in vacant_vms] if tpu is not None]
            _, tpu_snapshot = self.file_state.snapshot()
            tracked_names = set(tpu_snapshot.keys())
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
            tpu, started_at = self._steal_target
            # Abort if the steal target no longer matches any queued job
            if not tpu_matches_job(tpu, job):
                thread_log(f"Steal target {tpu.name} no longer matches the next unmatched job. Aborting steal.")
                self._steal_target = None
                return
            info = check_vacancy(tpu.name, tpu.zone)
            elapsed = time.time() - started_at
            if not info["vacant"]:
                thread_log(f"TPU {tpu.name} became busy ({int(elapsed)}s / {self.steal_wait}s). Aborting steal.")
                self._steal_target = None
                return
            if elapsed < self.steal_wait:
                thread_log(f"TPU {tpu.name} still vacant ({int(elapsed)}s / {self.steal_wait}s)")
                return

            # Wait complete; hand off to init workers
            thread_log(f"Stealing TPU {tpu.name}. Queued for init.", force_print=True)
            with self.file_state.transact():
                if tpu.name in self.file_state.tpus:
                    thread_log(f"Skipping steal for {tpu.name}: already tracked.")
                else:
                    self.file_state.tpus[tpu.name] = ManagedTPU(tpu=tpu, owned=False, status="need_init")
                    # Prevent the same job from being matched again
                    reserved_job = self.file_state.jobs.get(job.job_id)
                    if reserved_job is not None and reserved_job.status == "queued":
                        reserved_job.assigned_tpu = tpu
            self._steal_target = None


    def run_job(self, job_id: int, tpu_name: str):
        """Will be run in a new thread."""
        log_path = self.log_dir / f"job_{job_id}.log"
        fs = self.file_state
        launch_tpu: TPU | None = None
        command = ""
        stage_dir = ""
        run_name = ""
        project_name = ""
        attempt = 0
        max_retry = 0
        with fs.transact():
            job = fs.jobs.get(job_id)
            if job is None:
                raise ValueError(f"run_job: Job {job_id} not found in file state")
            mt = fs.tpus.get(tpu_name)
            if mt is None:
                raise ValueError(f"run_job: ManagedTPU {tpu_name} not found in file state")
            if job.status == "cancelled":
                if mt.status == "busy":
                    mt.status = "free"
                thread_log(f"[job {job_id}] skipping launch: job is cancelled", force_print=True)
                return
            if job.status != "running":
                thread_log(f"[job {job_id}] skipping launch: job is {job.status}, expected running", force_print=True)
                return
            if job.assigned_tpu is None or job.assigned_tpu.name != tpu_name:
                thread_log(f"[job {job_id}] skipping launch: assigned TPU mismatch", force_print=True)
                return
            if mt.status != "busy":
                thread_log(f"[job {job_id}] skipping launch: TPU {tpu_name} is {mt.status}, expected busy", force_print=True)
                return

            launch_tpu = mt.tpu
            command = job.command
            stage_dir = job.stage_dir
            run_name = job.run_name
            project_name = job.project_name
            attempt = job.attempt
            max_retry = job.max_retry

        with open(log_path, "a") as log_file, set_thread_vars(log_file=log_file):
            try:
                assert launch_tpu is not None
                thread_log(f"[job {job_id}] Launching on {launch_tpu.name}...")
                thread_log(f"[job {job_id}] See job log at {log_path}")

                returncode = launch(
                    launch_tpu, command, stage_dir=stage_dir,
                    run_name=run_name, project_name=project_name
                )

                # Restage code if retry
                might_retry = (
                    max_retry > attempt
                    and returncode != 0
                    and returncode != EXIT_CODE_FATAL
                )
                new_stage_dir = None
                if might_retry:
                    new_stage_dir = stage_code(run_name, project_name)

                tpu_dead = False
                if returncode == EXIT_CODE_SSH_RETRY:  # this means the TPU might be dead
                    exists, state = gcloud_describe_tpu(launch_tpu.name, launch_tpu.zone)
                    if not exists or state != "READY":
                        thread_log(f"[job {job_id}] TPU {launch_tpu.name} failed mid-task, removing it.", force_print=True)
                        tpu_dead = True

                with fs.transact():
                    job = fs.jobs[job_id]   # Implicit: no other processes can touch this job
                    mt = fs.tpus.get(tpu_name)  # Careful: the TPU might be nonexistent!
                    if mt is not None:
                        mt.status = "free"  # Free it first
                        if tpu_dead:
                            fs.tpus.pop(tpu_name, None)
                    if job.status == "cancelled":
                        thread_log(f"[job {job_id}] cancelled, skipping retry", force_print=True)
                    elif returncode == 0:
                        job.status = "succeeded"
                        thread_log(f"[job {job_id}] succeeded", force_print=True)
                    elif new_stage_dir is not None:  # might_retry
                        job.status = "queued"
                        job.assigned_tpu = None
                        job.priority += 1  # make-do logic
                        job.stage_dir = new_stage_dir
                        thread_log(f"[job {job_id}] failed and re-queued (exit {returncode})", force_print=True)
                    else:
                        job.status = "failed"
                        thread_log(f"[job {job_id}] failed (exit {returncode})", force_print=True)

            except Exception as e:
                thread_log(f"[job {job_id}] unhandled exception:\n{e}", force_print=True)
                with fs.transact():
                    job = fs.jobs[job_id]
                    if job.status != "cancelled":
                        job.status = "failed"
                    mt = fs.tpus.get(tpu_name)
                    if mt is not None:
                        mt.status = "free"
    
    def summary(self) -> str:
        fs = self.file_state
        jobs_snapshot, tpus_snapshot = fs.snapshot()

        n_queued_jobs = sum(1 for j in jobs_snapshot.values() if j.status == "queued")
        n_owned_tpus = sum(1 for mt in tpus_snapshot.values() if mt.owned)
        n_stolen_tpus = len(tpus_snapshot) - n_owned_tpus
        header = f"{n_owned_tpus} owned, {n_stolen_tpus} stolen, {n_queued_jobs} queued jobs"

        # Build per-TPU status lines
        tpu_to_job = {}
        for j in jobs_snapshot.values():
            if j.status == "running" and j.assigned_tpu:
                tpu_to_job[j.assigned_tpu.name] = j

        summary_lines = []
        for tpu_name, mt in tpus_snapshot.items():
            job = tpu_to_job.get(tpu_name)
            if job:
                summary_lines.append(f"  {mt.tpu.name}: running job {job.job_id} ({job.run_name})")
            else:
                summary_lines.append(f"  {mt.tpu.name}: {mt.status}")
        
        summary = header
        if summary_lines:
            summary += "\n" + "\n".join(summary_lines)
        return summary

    def run_tick(self, prev_tick: tuple|None):
        fs = self.file_state
        self.reap_threads()

        tick = self.summary()  # this does a file_state read
        if tick != prev_tick:
            thread_log(tick, force_print=True)

        matched_this_round: set[str] = set()
        matched_pairs: list[tuple[int, str, str]] = []
        first_unmatched: Job | None = None

        with fs.transact():
            stolen_names = {tpu_name for tpu_name, mt in fs.tpus.items() if not mt.owned}
            stolen_running = sum(
                1 for j in fs.jobs.values()
                if j.status == "running" and j.assigned_tpu
                and j.assigned_tpu.name in stolen_names
            )
            queued_jobs = [
                j for j in self.file_state.jobs.values()
                if j.status == "queued"
            ]
            queued_jobs.sort(key=lambda j: (-j.priority, j.created_at))

            for job in queued_jobs:
                if job.status != "queued":
                    continue
                if job.assigned_tpu is not None:  # Logic to prevent from matching the same job to different machines
                    assigned_mt = fs.tpus.get(job.assigned_tpu.name)
                    if (
                        assigned_mt is None
                        or not tpu_matches_job(assigned_mt.tpu, job)
                    ):
                        # Reservation is stale or invalid; fall back to normal matching.
                        job.assigned_tpu = None
                    elif assigned_mt.status == "free":
                        if assigned_mt.tpu.name in matched_this_round:
                            continue
                        if (
                            not assigned_mt.owned
                            and self.max_steal >= 0
                            and stolen_running >= self.max_steal
                        ):
                            continue
                        assigned_mt.status = "busy"
                        job.status = "running"
                        job.assigned_tpu = assigned_mt.tpu
                        job.attempt += 1
                        matched_this_round.add(assigned_mt.tpu.name)
                        if not assigned_mt.owned:
                            stolen_running += 1
                        matched_pairs.append((job.job_id, assigned_mt.tpu.name, job.run_name))
                        continue
                    else:
                        # busy/need_init/initializing: keep waiting on this reservation.
                        continue

                match = self.match_tpu(job, fs.tpus, exclude=matched_this_round)
                # If matched a stolen TPU, check budget
                if match and not match.owned:
                    if self.max_steal >= 0 and stolen_running >= self.max_steal:
                        match = None
                if match:
                    matched_mt = fs.tpus.get(match.tpu.name)
                    if matched_mt is None or matched_mt.status != "free":
                        continue
                    matched_mt.status = "busy"
                    job.status = "running"
                    job.assigned_tpu = matched_mt.tpu
                    job.attempt += 1
                    matched_this_round.add(match.tpu.name)
                    if not match.owned:
                        stolen_running += 1
                    matched_pairs.append((job.job_id, match.tpu.name, job.run_name))
                elif first_unmatched is None:
                    first_unmatched = job

        for job_id, tpu_name, run_name in matched_pairs:
            thread_log(f"Matched job {job_id} ({run_name}) -> {tpu_name}", force_print=True)
            t = threading.Thread(target=self.run_job, args=(job_id, tpu_name))
            t.start()
            self._job_threads[job_id] = t

        # Try stealing for first unmatched job if budget allows
        if first_unmatched and self.steal_wait >= 0:
            if self.max_steal < 0 or stolen_running < self.max_steal:
                self.steal_tick(first_unmatched)

        self._stop_event.wait(self.tick_interval)
        if self._stop_file.exists():
            thread_log("Stop file detected, shutting down...", force_print=True)
            self._stop_event.set()
        
        return tick

    def _join_job_threads(self, timeout: float = 10.0):
        for jid, t in list(self._job_threads.items()):
            t.join(timeout=timeout)
            if not t.is_alive():
                self._job_threads.pop(jid, None)
            else:
                thread_log(f"Thread for job {jid} did not exit within {timeout}s", force_print=True)

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
                self._join_job_threads()
                self.reap_threads()
