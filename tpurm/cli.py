import argparse
import time
from pathlib import Path
from typing import Optional, get_args

from .common import thread_log, DatasetName
from .scheduler import FILE_STATE_DIR, Scheduler
from .steal import scan_target, get_zone_from_name
from .staging import stage_code, kill_remote_processes
from .state import FileState, Job
from .freeze import freeze


def submit_job(
    tpu_size: list[str],
    region: list[str]|None,
    run_name: str,
    project_name: str,
    command: Optional[str],
    command_path: Optional[str],
    datasets: list[DatasetName],
    priority: int,
    max_att: int,
    state_dir: Path = FILE_STATE_DIR,
) -> int:
    """Appends new jobs in queued state."""
    assert (command_path is None) ^ (command is None), "Exactly one of command and command_path must be provided"
    if command is None:
        command = Path(command_path).read_text().strip()  # type: ignore
    if command.startswith("python "):
        raise ValueError("Use python3.13 instead of python.")
    freeze()
    stage_dir = stage_code(run_name, project_name)

    fs = FileState(state_dir)
    with fs.transact():
        job_id = fs._next_job_id
        fs._next_job_id += 1
        fs._jobs[job_id] = Job(
            job_id=job_id,
            created_at=time.time(),
            command=command,
            tpu_size=tpu_size,
            region=region,
            datasets=datasets,
            run_name=run_name,
            project_name=project_name,
            stage_dir=stage_dir,
            attempt=1,
            max_att=max_att,
            priority=priority,
            assigned_tpu=None,
            status="queued",
        )
    thread_log(f"Submitted job {job_id}: run_name={run_name}")
    return job_id


def resume_job(job_id: int, state_dir: Path = FILE_STATE_DIR):
    fs = FileState(state_dir)
    with fs.transact():
        if job_id not in fs._jobs:
            raise ValueError(f"Job {job_id} not found.")
        job = fs._jobs[job_id]
        job.status = "queued"
        job.attempt = 1
        if job.assigned_tpu is not None:
            job.region = [job.assigned_tpu.region]
    thread_log(f"Resumed job {job_id} at region {job.region}")


def cancel_job(job_id: int, state_dir: Path = FILE_STATE_DIR):
    fs = FileState(state_dir)

    with fs.transact():
        if job_id not in fs._jobs:
            thread_log(f"Error: Job {job_id} not found.")
            return
        job = fs._jobs[job_id]
        if job.status == "queued":
            job.assigned_tpu = None
            job.status = "cancelled"
            thread_log(f"Cancelled queued job {job_id}")
        elif job.status in ("matched", "running"): 
            thread_log(f"Marked {job.status} job {job_id} as cancelled.")
            job.status = "cancelled"
        else:
            thread_log(f"Error: Job {job_id} is already {job.status}.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TPU Job Scheduler")
    subparsers = parser.add_subparsers(dest="action")

    # submit
    sub = subparsers.add_parser("submit", help="Submit a job")
    sub.add_argument("--tpu-size", nargs="+", required=True)
    sub.add_argument("--region", nargs="+", default=None)
    sub.add_argument("--run-name", required=True)
    sub.add_argument("--project-name", required=True)
    sub.add_argument("--command", default=None)
    sub.add_argument("--command-path", default=None)
    sub.add_argument("--dataset", nargs="+", choices=list(get_args(DatasetName)), required=True)
    sub.add_argument("--priority", type=int, default=0, help="Priority of the job")
    sub.add_argument("--max-att", type=int, default=0, help="Max number of attempts (including first run)")

    # resume
    sub = subparsers.add_parser("resume", help="Resume a job")
    sub.add_argument("job_id", type=int)

    # cancel
    sub = subparsers.add_parser("cancel", help="Cancel a job")
    sub.add_argument("job_id", type=int)

    # kill
    sub = subparsers.add_parser("kill", help="Kill remote TPU processes by TPU name")
    sub.add_argument("tpu_name")
    
    # freeze
    sub = subparsers.add_parser("freeze", help="Freeze the environment")
    
    # scan
    sub = subparsers.add_parser("scan", help="Scan for vacant TPUs")
    sub.add_argument("--tpu-size", nargs="+", required=True)
    sub.add_argument("--region", nargs="+", required=True)

    # daemon
    sub = subparsers.add_parser("start", help="Run the scheduler daemon")
    sub.add_argument("--steal-wait", type=int, default=-1, help="Seconds to wait before stealing (-1 to disable)")
    sub.add_argument("--steal-max", type=int, default=2, help="Max jobs on stolen TPUs (-1 for unlimited)")
    sub.add_argument("--alloc-max", type=int, default=4, help="Target number of owned TPUs (0 disables allocation)")
    sub.add_argument("--alloc-sizes", nargs="+", default=["v6e-64", "v5p-64"], help="TPU sizes to allocate")
    sub.add_argument("--alloc-regions", nargs="+", default=None, help="Restrict allocation to these regions")
    sub.add_argument("--alloc-workers", type=int, default=4, help="Parallel allocation workers")
    sub.add_argument("--init-workers", type=int, default=4, help="Parallel initialization workers")

    # stop
    subparsers.add_parser("stop", help="Gracefully stop the daemon")

    args = parser.parse_args(argv)

    if args.action == "submit":
        if (args.command is None) == (args.command_path is None):
            parser.error("Provide exactly one of --command or --command-path")
        run_name = time.strftime("%y%m%d%H%M%S") + "-" + args.run_name
        submit_job(
            tpu_size=args.tpu_size,
            region=args.region,
            run_name=run_name,
            project_name=args.project_name,
            command=args.command,
            command_path=args.command_path,
            datasets=args.dataset,
            priority=args.priority,
            max_att=args.max_att,
        )
    elif args.action == "resume":
        resume_job(args.job_id)
    elif args.action == "cancel":
        cancel_job(args.job_id)
    elif args.action == "kill":
        zone = get_zone_from_name(args.tpu_name)
        return 0 if kill_remote_processes(args.tpu_name, zone, "/tmp/tpurm-kill-all-no-log-dir") else 1
    elif args.action == "scan":
        vacant = scan_target(args.tpu_size, args.region)
        thread_log(f"\n{len(vacant)} vacant TPU(s) found.", force_print=True)
    elif args.action == "freeze":
        freeze()
    elif args.action == "start":
        scheduler = Scheduler(
            alloc_max=args.alloc_max,
            alloc_sizes=args.alloc_sizes,
            alloc_regions=args.alloc_regions,
            alloc_workers=args.alloc_workers,
            init_workers=args.init_workers,
            steal_wait=args.steal_wait,
            steal_max=args.steal_max,
        )
        scheduler.run()

    elif args.action == "stop":
        stop_file = FILE_STATE_DIR / "tpurm.stop"
        stop_file.touch()
        thread_log(f"Stop file created: {stop_file}", force_print=True)
    else:
        parser.print_help()
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
