import argparse
import shlex
import shutil
import time
from pathlib import Path
from typing import get_args

from .freeze import freeze
from .globals import DatasetName
from .filestate import FILE_STATE_DIR, Filestate, Job
from .scheduler import Scheduler, allocation_combos
from .staging import stage_code, stage_dir_to_log_dir, stage_dir_to_log_root
from .steal import scan_target
from .tpu import REGION_BUCKETS
from .util_log import LogContext
from .util_ssh import kill_remote_processes

def infer_resume_region(resume_from: str) -> str:
    for region, bucket in REGION_BUCKETS.items():
        if resume_from == bucket or resume_from.startswith(bucket + "/"):
            return region
    raise ValueError(f"Could not infer region from resume checkpoint: {resume_from}")


def submit_job(
    tpu_size: list[str],
    region: list[str]|None,
    run_name: str,
    project_name: str,
    command: str | None,
    command_path: str | None,
    resume_from: str | None,
    datasets: list[DatasetName],
    priority: int,
    max_att: int,
    *,
    log_ctx: LogContext,
    state_dir: Path = FILE_STATE_DIR,
) -> int:
    """Appends new jobs in queued state."""
    assert (command_path is None) ^ (command is None), "Exactly one of command and command_path must be provided"
    if command is None:
        command = Path(command_path).read_text().strip()  # type: ignore
    if command.startswith("python "):
        raise ValueError("Use python3.13 instead of python.")
    if resume_from is not None:
        resume_region = infer_resume_region(resume_from)
        if region != [resume_region]:
            log_ctx.log(f"Pinning job to region {resume_region} for resume checkpoint {resume_from}")
        region = [resume_region]
    if len(allocation_combos(tpu_size, region)) == 0:
        raise ValueError("No valid TPU size/region combination is possible for this job request.")
    freeze()
    stage_dir = stage_code(run_name, project_name, log_ctx=log_ctx)
    stored_max_att = None if max_att <= 0 else max_att

    fs = Filestate(state_dir)
    with fs.transact():
        job_id = len(fs._jobs)
        fs._jobs.append(Job(
            id=job_id,
            created_at=time.time(),
            command=command,
            resume_from=resume_from,
            tpu_size=tpu_size,
            region=region,
            datasets=datasets,
            run_name=run_name,
            project_name=project_name,
            stage_dir=stage_dir,
            attempt=1,
            max_att=stored_max_att,
            priority=priority,
            assigned_tpu=None,
            status="queued",
        ))
    log_ctx.log(f"Submitted job {job_id}: run_name={run_name}")
    return job_id


def resume_job(job_id: int, *, log_ctx: LogContext, state_dir: Path = FILE_STATE_DIR):
    fs = Filestate(state_dir)
    with fs.transact():
        if not (0 <= job_id < len(fs._jobs)):
            raise ValueError(f"Job {job_id} not found.")
        job = fs._jobs[job_id]
        log_root = Path(stage_dir_to_log_root(job.stage_dir))
        log_dir = Path(stage_dir_to_log_dir(job.stage_dir, attempt=1))
        if log_root.exists():
            try:
                shutil.rmtree(log_root)
            except PermissionError as exc:
                sudo_rm = f"sudo rm -rf {shlex.quote(str(log_root))}"
                recreate = (
                    f"mkdir -p {shlex.quote(str(log_dir))} && "
                    f"chmod 777 {shlex.quote(str(log_dir))}"
                )
                raise PermissionError(
                    "Could not clear existing logs during resume.\n"
                    f"Run:\n  {sudo_rm}\n  {recreate}\n"
                    f"Then rerun: tpurm resume {job_id}"
                ) from exc
            log_ctx.log(f"Cleared existing logs at {log_root}")
        log_dir.mkdir(parents=True, exist_ok=True)
        job.status = "queued"
        job.attempt = 1
        job.log_dir = stage_dir_to_log_dir(job.stage_dir, attempt=job.attempt)
        if job.assigned_tpu is not None:
            job.region = [job.assigned_tpu.region]
    log_ctx.log(f"Resumed job {job_id} at region {job.region}")


def cancel_job(job_id: int, *, log_ctx: LogContext, state_dir: Path = FILE_STATE_DIR):
    fs = Filestate(state_dir)

    with fs.transact():
        if not (0 <= job_id < len(fs._jobs)):
            log_ctx.log(f"Error: Job {job_id} not found.")
            return
        job = fs._jobs[job_id]
        if job.status == "queued":
            job.assigned_tpu = None
            job.status = "cancelled"
            log_ctx.log(f"Cancelled queued job {job_id}")
        elif job.status in ("waiting", "running"):
            log_ctx.log(f"Cancelled {job.status} job {job_id}.")
            job.status = "cancelled"
        else:
            log_ctx.log(f"Error: Job {job_id} is already {job.status}.")


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
    sub.add_argument("--resume-from", default=None)
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
    sub.add_argument("tpu_name", type=str)
    sub.add_argument("--zone", type=str, required=True)
    
    # freeze
    sub = subparsers.add_parser("freeze", help="Freeze the environment")
    
    # scan
    sub = subparsers.add_parser("scan", help="Scan for vacant TPUs")
    sub.add_argument("--tpu-size", nargs="+", required=True)
    sub.add_argument("--region", nargs="+", required=True)

    # daemon
    sub = subparsers.add_parser("start", help="Run the scheduler daemon")
    sub.add_argument("--steal-wait", type=int, default=300, help="-1 to disable stealing")
    sub.add_argument("--steal-max", type=int, default=-1, help="-1 for unlimited stealing")
    sub.add_argument("--alloc-max", type=int, default=6)
    sub.add_argument("--alloc-sizes", nargs="+", default=["v6e-64", "v5p-64"])
    sub.add_argument("--alloc-regions", nargs="+", default=None)
    sub.add_argument("--alloc-workers", type=int, default=None, help="Defaults to alloc_max")
    sub.add_argument("--init-workers", type=int, default=None, help="Defaults to alloc_max")
    sub.add_argument("--forbid-owner", nargs="+", default=[], help="Do not match or steal TPUs owned by these users")

    # stop
    subparsers.add_parser("stop", help="Gracefully stop the daemon")

    args = parser.parse_args(argv)
    log_ctx = LogContext(None)

    if args.action == "submit":
        if (args.command is None) == (args.command_path is None):
            parser.error("Provide exactly one of --command or --command-path")
        if len(allocation_combos(args.tpu_size, args.region)) == 0:
            parser.error("No valid TPU size/region combination is possible for this job request.")
        run_name = time.strftime("%y%m%d%H%M%S") + "-" + args.run_name
        submit_job(
            tpu_size=args.tpu_size,
            region=args.region,
            run_name=run_name,
            project_name=args.project_name,
            command=args.command,
            command_path=args.command_path,
            resume_from=args.resume_from,
            datasets=args.dataset,
            priority=args.priority,
            max_att=args.max_att,
            log_ctx=log_ctx,
        )
    elif args.action == "resume":
        resume_job(args.job_id, log_ctx=log_ctx)
    elif args.action == "cancel":
        cancel_job(args.job_id, log_ctx=log_ctx)
    elif args.action == "kill":
        return 0 if kill_remote_processes(
            args.tpu_name,
            args.zone,
            "/tmp/tpurm-kill-all-no-log-dir",
            log_ctx=log_ctx,
        ) else 1
    elif args.action == "scan":
        vacant = scan_target(args.tpu_size, args.region, log_ctx=log_ctx)
        log_ctx.log(f"{len(vacant)} vacant TPU(s) found.")
    elif args.action == "freeze":
        freeze()
    elif args.action == "start":
        if len(allocation_combos(args.alloc_sizes, args.alloc_regions)) == 0:
            parser.error("No valid allocation size/region combination is possible.")
        if args.alloc_workers is None:
            args.alloc_workers = args.alloc_max
        if args.init_workers is None:
            args.init_workers = args.alloc_max
        scheduler = Scheduler(
            alloc_max=args.alloc_max,
            alloc_sizes=args.alloc_sizes,
            alloc_regions=args.alloc_regions,
            alloc_workers=args.alloc_workers,
            init_workers=args.init_workers,
            steal_wait=args.steal_wait,
            steal_max=args.steal_max,
            forbidden_owners=set(args.forbid_owner),
        )
        scheduler.run()

    elif args.action == "stop":
        FILE_STATE_DIR.mkdir(parents=True, exist_ok=True)
        stop_file = FILE_STATE_DIR / "tpurm.stop"
        stop_file.touch()
        log_ctx.log(f"Stop file created: {stop_file}")
    else:
        parser.print_help()
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
