import json
import os
import fcntl
import copy
import threading
from pathlib import Path
from typing import Any, Literal
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from .common import TPU, DatasetName, REPO_ROOT
from .staging import stage_dir_to_log_dir

FILE_STATE_DIR = REPO_ROOT / ".tpurm"

JobStatus = Literal["queued", "matched", "running", "succeeded", "failed", "cancelled", "orphaned"]
TPUStatus = Literal["need_init", "initializing", "free", "busy"]

@dataclass
class Job:
    # Immutable properties, known at creation time
    job_id: int
    created_at: float
    command: str
    tpu_size: list[str]
    region: list[str] | None
    datasets: list[DatasetName]  # datasets required by the script
    run_name: str
    project_name: str
    stage_dir: str
    log_dir: str=field(init=False)
    attempt: int
    max_att: int
    priority: int  # higher is more prioritized

    # Mutable properties: can be changed by scheduler
    assigned_tpu: TPU | None
    status: JobStatus

    def __post_init__(self):
        self.log_dir = stage_dir_to_log_dir(self.stage_dir)


@dataclass
class ManagedTPU:
    tpu: TPU
    owned: bool
    status: TPUStatus
    datasets: list[DatasetName]=field(default_factory=list)  # datasets that TPU has mounted


class FileState:
    """
    File-based state for communication.

    Contract:
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

    @contextmanager
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
        self._jobs = {
            int(jid): Job(**{
                k: _deserialize(v)
                for k, v in job.items()
                if k != "log_dir"
            })
            for jid, job in data["jobs"].items()
        }
        self._tpus = {tpu_name: ManagedTPU(**{k: _deserialize(v) for k, v in tpu.items()}) for tpu_name, tpu in data["tpus"].items()}

    def _save_unlocked(self):
        data = {
            "next_job_id": self._next_job_id,
            "jobs": {
                str(jid): {
                    k: v
                    for k, v in asdict(job).items()
                    if k != "log_dir"
                }
                for jid, job in self._jobs.items()
            },
            "tpus": {tpu_name: asdict(tpu) for tpu_name, tpu in self._tpus.items()},
        }
        tmp = str(self._file_path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self._file_path)


# helper for loading file state
def _deserialize(d: Any):
    if isinstance(d, dict) and all(k in d for k in ["size", "mode", "owner", "id", "zone"]):
        tpu = TPU(d["size"], d["mode"], d["owner"], d["id"], d["zone"])
        if d.get("name") and d["name"] != tpu.name:
            tpu.name = d["name"]
        if "num_workers" in d:
            tpu.num_workers = d["num_workers"]
        return tpu
    else:
        return d
