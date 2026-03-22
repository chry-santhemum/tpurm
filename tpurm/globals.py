import threading
from dotenv import dotenv_values
from pathlib import Path
from typing import Literal

ENV_VARS = dotenv_values(Path.home() / ".env")
_THREAD_LOCAL = threading.local()
_THREAD_VAR_MISSING = object()

REMOTE_SCRIPTS_DIR = Path(__file__).resolve().parent / "remote"

def resolve_repo_root() -> Path:
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError(f"Could not find git repo root: {cwd}")

REPO_ROOT = resolve_repo_root()
NFS_US = "/kmh-nfs-us-mount"
NFS_SSD_US = "/kmh-nfs-ssd-us-mount"
DEFAULT_SA_KEY_FILE: str = ENV_VARS["DEFAULT_SA_KEY_FILE"]  # type: ignore
DEFAULT_KEYS_DIR: str = ENV_VARS["DEFAULT_KEYS_DIR"]  # type: ignore
DatasetName = Literal["imagenet", "fineweb10B"]
