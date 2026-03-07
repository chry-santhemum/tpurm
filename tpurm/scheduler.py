"""Lightweight TPU job scheduler."""

# Agents: Please DO NOT implement these functionalities unless told explicitly to do so.
# TODO: Feature list
# Build tests that are actually based on simulating the finite state machine
# If there are jobs queued, allocator should try to allocate according to the jobs queued, not randomly
# Match jobs down the list as well, not eagerly
# Call freeze when staging code / when to install dependencies
# Make common TPU ssh commands simple CLI, e.g. "tpu ssh --name ... --command ..."
# Delete Bucket folder after failure
# Relax restart constraint when job failed and no checkpoints saved
# Make allocator/schedule hparams part of the file state, hence modifiable mid-run

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
    TPU, REPO_ROOT, TPU_CONFIGS, NFS_SSD_US,
    thread_log, set_thread_vars, zone_to_region, name_to_tpu, size_to_family,
    gcloud_delete_tpu, gcloud_describe_tpu, list_tpus, check_data_mount, check_vacancy,
)
from .staging import (
    stage_code, launch, poll_launch, has_fatal_error_in_logs,
    kill_remote_processes, EXIT_CODE_FATAL, EXIT_CODE_SSH_RETRY,
)
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
    max_retry: int  # max attempts (0 means one execution attempt)
    # Mutable properties
    status: JobStatus
    assigned_tpu: TPU | None  # last assigned
    attempt: int
    priority: int  # higher is more prioritized
    log_dir: str | None = None

    @property
    def launch_token(self) -> str:
        return self.stage_dir.split("/")[-1].split("__")[-2]

def _deserialize(d: Any):
    # convenience for loading file state
    if isinstance(d, dict) and all(k in d for k in ["size", "mode", "owner", "id", "zone"]):
        tpu = TPU(d["size"], d["mode"], d["owner"], d["id"], d["zone"])
        if d.get("name") and d["name"] != tpu.name:
            tpu.name = d["name"]
        if "num_workers" in d:
            tpu.num_workers = d["num_workers"]
        return tpu
    else:
        return d

class FileState:
    """
    File-based state for communication.

    Rules:
    - submit() will only ADD new jobs in "queued" state.
    - cancel() will only change the state of jobs to "cancelled" state.
    - alloc workers will only ADD new ManagedTPUs with "need_init" status.
    - init workers will only change:
        - "need_init" to "initializing"
        - "initializing" to "free" if successful
        - Removing "initializing" TPUs
    """
    _next_job_id: int
    _jobs: dict[int, Job]  # job_id -> Job
    _tpus: dict[str, ManagedTPU]  # tpu_name -> ManagedTPU

    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        state_dir.mkdir(parents=True, exist_ok=True)

        self._file_path = state_dir / "state.json"
        self._lock_path = state_dir / "state.lock"
        self._mutex = threading.RLock()
    
    # Public, thread-safe methods
    def snapshot(self) -> tuple[int, dict[int, Job], dict[str, ManagedTPU]]:
        """Return a deepcopy of the file content."""
        with self._mutex, open(self._lock_path, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_SH)
            self._load_unlocked()
            return (self._next_job_id, copy.deepcopy(self._jobs), copy.deepcopy(self._tpus))

    @contextlib.contextmanager
    def transact(self):
        """Read and write."""
        with self._mutex, open(self._lock_path, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            self._load_unlocked()
            yield self
            self._save_unlocked()

    # These methods are NOT thread-safe
    def _load_unlocked(self):
        if not self._file_path.exists():
            self._next_job_id = 1
            self._jobs = {}
            self._tpus = {}
            return
        with open(self._file_path) as f:
            data = json.load(f)
        self._next_job_id = data["next_job_id"]
        self._jobs = {int(jid): Job(**{k: _deserialize(v) for k, v in job.items()}) for jid, job in data["jobs"].items()}
        self._tpus = {tpu_name: ManagedTPU(**{k: _deserialize(v) for k, v in tpu.items()}) for tpu_name, tpu in data["tpus"].items()}

    def _save_unlocked(self):
        data = {
            "next_job_id": self._next_job_id,
            "jobs": {str(jid): asdict(job) for jid, job in self._jobs.items()},
            "tpus": {tpu_name: asdict(tpu) for tpu_name, tpu in self._tpus.items()},
        }
        tmp = str(self._file_path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self._file_path)


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


def stage_dir_to_log_dir(stage_dir: str) -> str:
    stage_dir_suffix = stage_dir.split("/staging/")[-1]
    return f"{NFS_SSD_US}/logs/{stage_dir_suffix}"

# TODO: Make the slow network requests parallelized
def sync_tracked_tpus(file_state: FileState, startup: bool=False):
    """
    Scan and sync TPU status.
    Only modifies the file_state._tpus dict.

    Args:
        startup: If true, scan all zones to check for TPUs with my name.
    """
    update_status: dict[tuple[str, str], str] = {}  # (tpu_name, zone) -> status
    update_num_workers: dict[tuple[str, str], int] = {}
    read_status: dict[tuple[str, str], str] = {}  # Status at snapshot time
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

    _, jobs, tpus = file_state.snapshot()
    running_tpu_names = {
        j.assigned_tpu.name for j in jobs.values()
        if j.status == "running" and j.assigned_tpu
    }
    for tpu_name, mt in tpus.items():
        read_status[(tpu_name, mt.tpu.zone)] = mt.status
    for tpu_name, mt in tpus.items():
        key = (tpu_name, mt.tpu.zone)
        exists = None
        if startup:
            exists, _, num_workers = gcloud_describe_tpu(*key)
            if num_workers is not None:
                update_num_workers[key] = num_workers
        if mt.status in ["initializing", "need_init"]:
            if exists is None:
                exists, _, _ = gcloud_describe_tpu(*key)
            if not exists:
                thread_log(f"{mt.tpu.name} was preempted during initialization. Untracking.")
                update_status[key] = "untrack"
            continue  # Don't change the status otherwise
        if mt.owned and (mt.tpu.name not in [t.name for t in owned]):
            owned.append(mt.tpu)
        # Stolen TPUs
        elif (check_data_mount(*key) and check_env(*key)):
            if (not check_vacancy(*key)["vacant"]) or (mt.tpu.name in running_tpu_names):
                thread_log(f"Previously-stolen TPU {mt.tpu.name} has a mount but is busy.")
                update_status[key] = "busy"
            else:
                thread_log(f"Previously-stolen TPU {mt.tpu.name} seems to be free.")
                update_status[key] = "free"
        else:
            thread_log(f"Previously-stolen TPU {mt.tpu.name} doesn't have the env anymore. Untracking it.")
            update_status[key] = "untrack"

    # Owned TPUs
    for tpu in owned:
        key = (tpu.name, tpu.zone)
        exists, _, num_workers = gcloud_describe_tpu(*key)
        if num_workers is not None:
            tpu.num_workers = num_workers
            update_num_workers[key] = num_workers
        if not exists:
            thread_log(f"Owned TPU {tpu.name} ({tpu.zone}) no longer exists.")
            update_status[key] = "untrack"
            continue
        if tpu.name in tpus:
            mt = tpus[tpu.name]
            if mt.status in ["initializing", "need_init"]:
                continue
        if (not check_vacancy(*key)["vacant"]) or (tpu.name in running_tpu_names):
            update_status[key] = "busy"
        elif check_data_mount(*key) and check_env(*key):
            update_status[key] = "free"
        else:
            update_status[key] = "need_init"

    # Write the file state
    with file_state.transact():
        to_untrack = set()
        for tpu_name, mt in file_state._tpus.items():
            key = (tpu_name, mt.tpu.zone)
            if key in update_num_workers:
                mt.tpu.num_workers = update_num_workers[key]
            if mt.status != read_status.get(key) or key not in update_status:
                continue
            elif update_status[key] == "untrack":
                to_untrack.add(tpu_name)
            else:
                mt.status = update_status[key]
        for tpu_name in to_untrack:
            del file_state._tpus[tpu_name]

        # Add TPUs not in the file state
        tracked_names = list(file_state._tpus.keys())
        for (name, zone), status in update_status.items():
            if name not in tracked_names and status != "untrack":
                tpu = name_to_tpu(name, zone)
                if tpu is None:
                    continue
                if (name, zone) in update_num_workers:
                    tpu.num_workers = update_num_workers[(name, zone)]
                file_state._tpus[tpu.name] = ManagedTPU(tpu=tpu, owned=True, status=status)


def submit_job(
    tpu_size: list[str], region: list[str]|None,
    run_name: str, project_name: str,
    command: Optional[str], command_path: Optional[str],
    priority: int = 0, max_retry: int = 0,
    state_dir: Path = Path(FILE_STATE_DIR),
) -> int:
    assert (command_path is None) ^ (command is None), "Exactly one of command and command_path must be provided"
    if command is None:
        command = Path(command_path).read_text().strip()  # type: ignore
    if command.startswith("python "):
        raise ValueError("Use python3.13 instead of python.")
    stage_dir = stage_code(run_name, project_name)

    fs = FileState(state_dir)
    with fs.transact():
        job_id = fs._next_job_id
        fs._next_job_id += 1
        fs._jobs[job_id] = Job(
            job_id=job_id,
            created_at=time.time(),
            command=command,
            stage_dir=stage_dir,
            tpu_size=tpu_size,
            region=region,
            run_name=run_name,
            project_name=project_name,
            max_retry=max_retry,
            status="queued",
            assigned_tpu=None,
            attempt=0,
            priority=priority,
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

# TODO: Place cancelled jobs in a queue to kill by the scheduler only
def cancel_job(job_id: int, state_dir: Path = Path(FILE_STATE_DIR)):
    fs = FileState(state_dir)

    tpu_to_kill = None
    with fs.transact():
        if job_id not in fs._jobs:
            thread_log(f"Job {job_id} not found.")
            return
        job = fs._jobs[job_id]
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
        with fs.transact():
            job = fs._jobs.get(job_id)
            if job is not None:
                job.assigned_tpu = None
            mt = fs._tpus.get(tpu_to_kill.name)
            if mt is not None:
                mt.status = "free"
            
        thread_log(f"Cancelled running job {job_id}")



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
        tick_interval: int = 30,
        state_dir: Path = Path(FILE_STATE_DIR),
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
        sync_tracked_tpus(fs, startup=True)

        with fs.transact():
            for tpu_name, mt in fs._tpus.items():
                if mt.status == "initializing":
                    thread_log(f"Resetting stuck initializing TPU {tpu_name} to need_init")
                    mt.status = "need_init"
            for job in fs._jobs.values():
                if job.status == "running" and job.log_dir is None:
                    job.log_dir = stage_dir_to_log_dir(job.stage_dir)

        self.poll_jobs()


    def try_alloc(self, worker_id: int, size: str, zone: str) -> bool:
        tpu = allocate(size, zone, max_retries=8, stop_events=[self._stop_event, self._alloc_sleep_event])
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

        thread_log(f"[worker {worker_id}] Initialization failed, rebooting: {tpu.name}", force_print=True)
        if not reboot(tpu):
            thread_log(f"[worker {worker_id}] Reboot failed, deleting: {tpu.name}", force_print=True)
            gcloud_delete_tpu(tpu.name, tpu.zone)
            with self.file_state.transact():
                self.file_state._tpus.pop(tpu.name, None)
            return False
        
        # Retry after reboot
        if ensure_ready(tpu, skip_upgrade=skip_upgrade):
            thread_log(f"[worker {worker_id}] Successfully initialized after reboot: {tpu.name}", force_print=True)
            with self.file_state.transact():
                self.file_state._tpus[tpu.name] = ManagedTPU(tpu=tpu, owned=mt.owned, status="free")
            return True

        thread_log(f"[worker {worker_id}] Initialization still failing after reboot, deleting: {tpu.name}", force_print=True)
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

    def match_tpu(self, job: Job, tpus: dict[str, ManagedTPU], exclude: set[str] | None = None) -> str | None:
        """Match a job to a free TPU, preferring owned over stolen."""
        if exclude is None:
            exclude = set()
        # Owned
        for tpu_name, mt in tpus.items():
            if mt.status != "free" or not mt.owned or tpu_name in exclude:
                continue
            if tpu_matches_job(mt.tpu, job):
                return tpu_name
        # Stolen
        for tpu_name, mt in tpus.items():
            if mt.status != "free" or mt.owned or tpu_name in exclude:
                continue
            if tpu_matches_job(mt.tpu, job):
                return tpu_name
        return None


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


    def steal_tick(self):
        """Scan for vacant TPUs to steal. After wait period, mark as need_init."""
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
            if job.status != "queued":  # The setter of job.status should also set job.assigned_tpu
                thread_log(f"Current steal target job {job.job_id} is no longer queued. Aborting steal.")
                self._steal_target = None
                return
            tpu, started_at = self._steal_target
            info = check_vacancy(tpu.name, tpu.zone)
            elapsed = time.time() - started_at
            if not info["vacant"]:
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
                    self.file_state._tpus[tpu.name] = ManagedTPU(tpu=tpu, owned=False, status="need_init")
            self._steal_target = None
            self._steal_job = None


    def launch_job(self, job: Job, mt: ManagedTPU):
        log_path = self.log_dir / f"job_{job.job_id}.log"
        with open(log_path, "a") as log_file, set_thread_vars(log_file=log_file):
            thread_log(f"[job {job.job_id}] Launching on {mt.tpu.name}...", force_print=True)

            returncode, log_dir = launch(
                mt.tpu, job.command, stage_dir=job.stage_dir,
                run_name=job.run_name, project_name=job.project_name
            )
            if returncode != 0:  # Job failed to start
                self.finalize_job(job, returncode)
                return
            
            fs = self.file_state
            with fs.transact():
                tracked_job = fs._jobs[job.job_id]
                if (
                    tracked_job.status == "running"
                    and tracked_job.assigned_tpu is not None
                    and tracked_job.assigned_tpu.name == mt.tpu.name
                ):
                    tracked_job.log_dir = log_dir


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
                max_retry = tracked_job.max_retry

            might_retry = (
                max_retry > attempt
                and returncode != 0
                and returncode != EXIT_CODE_FATAL
            )
            new_stage_dir = None
            if might_retry:
                new_stage_dir = stage_code(run_name, project_name)

            tpu_dead = False
            if returncode == EXIT_CODE_SSH_RETRY:
                exists, state, _ = gcloud_describe_tpu(tpu.name, tpu.zone)
                if not exists or state != "READY":
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
                    job.priority += 1
                    job.stage_dir = new_stage_dir
                    job.log_dir = None
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
            if job.log_dir is None:
                thread_log(f"[job {job.job_id}] ERROR: Job {job.job_id} has no log_dir.", force_print=True)
                continue

            returncode = poll_launch(job.log_dir, job.launch_token, job.assigned_tpu.num_workers)
            if returncode is None and job.status == "running":
                exists, state, _ = gcloud_describe_tpu(job.assigned_tpu.name, job.assigned_tpu.zone)
                if not exists or state != "READY":
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
        sync_tracked_tpus(self.file_state)

        tick, num_owned = self.summary()
        if tick != prev_tick:
            thread_log(tick, force_print=True)

        if num_owned >= self.alloc_max:
            self._alloc_sleep_event.set()
            thread_log(f"{num_owned}/{self.alloc_max} owned TPUs alive, alloc workers sleep")
        else:
            self._alloc_sleep_event.clear()
            thread_log(f"{num_owned}/{self.alloc_max} owned TPUs alive, alloc workers active")
        
        # Job matching
        matched_pairs: list[tuple[int, str]] = []  # (job_id, tpu_name)
        to_launch: list[tuple[int, str]] = []
        first_unmatched: Job | None = None
        
        fs = self.file_state
        with fs.transact():
            stolen_names = {tpu_name for tpu_name, mt in fs._tpus.items() if not mt.owned}
            stolen_running = sum(
                1 for j in fs._jobs.values()
                if j.status == "running" and j.assigned_tpu
                and j.assigned_tpu.name in stolen_names
            )
            queued_jobs = [
                j for j in fs._jobs.values()
                if j.status == "queued"
            ]
            queued_jobs.sort(key=lambda j: (-j.priority, j.created_at))
            
            # Matches jobs eagerly
            for job in queued_jobs:
                if job.assigned_tpu is not None:
                    assigned_name = job.assigned_tpu.name
                    assigned_mt = fs._tpus.get(assigned_name)
                    if assigned_mt is None:
                        job.assigned_tpu = None
                        if first_unmatched is None:
                            first_unmatched = job
                        continue
                    if not assigned_mt.owned:
                        if self.steal_max >= 0 and stolen_running >= self.steal_max:
                            if first_unmatched is None:
                                first_unmatched = job
                            continue
                        stolen_running += 1
                    matched_pairs.append((job.job_id, assigned_name))
                    continue
                match = self.match_tpu(job, fs._tpus, exclude={tpu_name for _, tpu_name in matched_pairs})
                # If matched a stolen TPU, check budget
                if match and not fs._tpus[match].owned:
                    if self.steal_max >= 0 and stolen_running >= self.steal_max:
                        match = None
                    else:
                        stolen_running += 1
                if match:
                    matched_pairs.append((job.job_id, match))
                elif first_unmatched is None:
                    first_unmatched = job

            for job_id, tpu_name in matched_pairs:
                matched_mt = fs._tpus[tpu_name]
                matched_job = fs._jobs[job_id]
                if matched_mt.status != "free":  # still initializing
                    thread_log(f"Matched TPU {tpu_name} is still in status {matched_mt.status}, skipping...")
                    continue
                matched_mt.status = "busy"
                matched_job.status = "running"
                matched_job.assigned_tpu = matched_mt.tpu
                matched_job.attempt += 1
                to_launch.append((matched_job.job_id, matched_mt.tpu.name))
        
        if self._stop_file.exists():
            thread_log("Stop file detected, shutting down...", force_print=True)
            self._stop_event.set()
        if self._stop_event.is_set():
            return tick

        # Launch jobs
        for job_id, tpu_name in to_launch:
            with fs.transact():
                job = fs._jobs[job_id]
                mt = fs._tpus[tpu_name]
                if (
                    job.status != "running"
                    or job.assigned_tpu is None
                    or job.assigned_tpu.name != tpu_name
                ):
                    continue
            self.launch_job(job, mt)

        # Try stealing for first unmatched job if budget allows
        if (
            first_unmatched is not None
            and self.steal_wait >= 0
            and (self.steal_max < 0 or stolen_running < self.steal_max)
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
