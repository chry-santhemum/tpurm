import json
import os
import fcntl
import threading
from pathlib import Path
from typing import Any, Literal
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict

from .globals import REPO_ROOT, DatasetName
from .tpu import TPU
from .staging import stage_dir_to_log_dir

FILE_STATE_DIR = REPO_ROOT / ".tpurm"

JobStatus = Literal["queued", "waiting", "running", "done", "cancelled"] 
# waiting: job has been matched to a TPU, but it needs initialization/warmup

TPUStatus = Literal["need_init", "initializing", "free", "busy"]

@dataclass
class Job:
    # Immutable
    id: int
    created_at: float
    command: str
    tpu_size: list[str]
    datasets: list[DatasetName]  # datasets required by the script
    run_name: str
    project_name: str
    stage_dir: str
    log_dir: str=field(init=False)
    max_att: int|None   # max number of attempts. None = unlimited
    priority: int       # higher is more prioritized

    # Mutable: can be changed by scheduler
    region: list[str]|None  # None = doesn't care
    attempt: int
    assigned_tpu: TPU|None
    status: JobStatus

    def __post_init__(self):
        self.log_dir = stage_dir_to_log_dir(self.stage_dir, attempt=self.attempt)

@dataclass
class ManagedTPU:
    tpu: TPU
    owned: bool
    status: TPUStatus
    datasets: list[DatasetName]=field(default_factory=list)  # datasets that TPU has mounted

class Filestate:
    """
    File-based state for communication between threads.
    """
    _jobs: list[Job]  # indexed by job.id
    _tpus: dict[str, ManagedTPU]  # tpu_name -> ManagedTPU

    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._file_path = self.state_dir / "state.json"
        self._lock_path = self.state_dir / "state.lock"
        self._mutex = threading.RLock()
    
    # Public, thread-safe methods
    def snapshot(self) -> tuple[list[Job], dict[str, ManagedTPU]]:
        """Return a deepcopy of the file content."""
        with self._mutex, open(self._lock_path, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_SH)
            self._load_unlocked()
            return (self._jobs, self._tpus)

    @contextmanager
    def transact(self):
        """Read and write. If an exception is raised, the file is not modified."""
        with self._mutex, open(self._lock_path, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            self._load_unlocked()
            yield self
            self._save_unlocked()

    # Private and NOT thread-safe
    def _load_unlocked(self):
        if not self._file_path.exists():
            self._jobs = []
            self._tpus = {}
            return
        with open(self._file_path) as f:
            data = json.load(f)
        self._jobs = [
            Job(**{k: _deserialize_maybe_tpu(v) for k, v in job.items() if k != "log_dir"})
            for job in data["jobs"]
        ]
        self._tpus = {
            tpu_name: ManagedTPU(**{k: _deserialize_maybe_tpu(v) for k, v in mt.items()})
            for tpu_name, mt in data["tpus"].items()
        }

    def _save_unlocked(self):
        data = {}
        data["jobs"] = [asdict(job) for job in self._jobs]
        data["tpus"] = {tpu_name: asdict(mt) for tpu_name, mt in self._tpus.items()}
        tmp = str(self._file_path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self._file_path)

# helper for loading filestate
def _deserialize_maybe_tpu(d: Any):
    """If d is dumped from a TPU, reconstruct it."""
    if isinstance(d, dict) and all(k in d for k in ["size", "mode", "owner", "id", "zone", "num_workers"]):
        tpu = TPU(
            size=d["size"], mode=d["mode"], owner=d["owner"], id=d["id"], zone=d["zone"], num_workers=d["num_workers"]
        )
        if d.get("name") and d["name"] != tpu.name:
            tpu.name = d["name"]
        return tpu
    return d
