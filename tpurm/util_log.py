import inspect
import os
import time
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_ROTATE_CHECK_EVERY = 100


def maybe_rotate_log(file: TextIO) -> None:
    """If log file is larger than `LOG_MAX_BYTES`, clear first half."""
    try:
        log_path = getattr(file, "name", None)
        if not isinstance(log_path, str):
            return
        file.flush()
        while Path(log_path).stat().st_size >= LOG_MAX_BYTES:
            with open(log_path, "r+") as rf:
                data = rf.read()
                if not data:
                    break
                keep = data[len(data) // 2:]
                nl = keep.find("\n")  # Find first newline to avoid partial line
                if nl != -1:
                    keep = keep[nl + 1:]
                rf.seek(0)
                rf.write(keep)
                rf.truncate()
                rf.flush()
        file.seek(0, os.SEEK_END)  # Move handle back to end of file
    except Exception:
        pass


@dataclass(slots=True)
class LogContext:
    """Handles log rotation."""
    file: TextIO
    _write_count: int = 0

    def log(self, msg: str, *, force_print: bool = False) -> None:
        ts = time.strftime("%y-%m-%d %H:%M:%S")
        caller = inspect.currentframe().f_back.f_code.co_name  # type: ignore
        line = f"[{ts}] [{caller}]: {msg}"
        self.file.write(line + "\n")
        self.file.flush()
        self._write_count += 1

        if self._write_count % LOG_ROTATE_CHECK_EVERY == 0:
            maybe_rotate_log(self.file)

        if force_print:
            print(line, flush=True)


def run_cmd(cmd: list[str], *, log_ctx: LogContext, **kwargs) -> subprocess.CompletedProcess:
    """
    `subprocess.run` wrapper with logging redirection.
    If `capture_output`, prefer using plain `subprocess.run` instead.
    """
    log_ctx.log(f"$ {shlex.join(cmd)}")
    kwargs.setdefault("text", True)
    if not kwargs.get("capture_output", False):
        kwargs.setdefault("stdout", log_ctx.file)
        kwargs.setdefault("stderr", log_ctx.file)
    result = subprocess.run(cmd, **kwargs)
    return result
