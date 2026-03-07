import argparse
import time
from pathlib import Path

from .common import thread_log
from .scheduler import FILE_STATE_DIR, Scheduler, cancel_job, submit_job
from .steal import scan_target
from .freeze import freeze


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TPU Job Scheduler")
    parser.add_argument(
        "--state-dir", type=Path, default=Path(FILE_STATE_DIR),
        help="Directory for state.json",
    )
    subparsers = parser.add_subparsers(dest="action")

    # submit
    sub = subparsers.add_parser("submit", help="Submit a job")
    sub.add_argument("--tpu-size", nargs="+", required=True)
    sub.add_argument("--region", nargs="+", default=None)
    sub.add_argument("--run-name", required=True)
    sub.add_argument("--project-name", default="jax-mae")
    sub.add_argument("--command", default=None)
    sub.add_argument("--command-path", default=None)
    sub.add_argument("--priority", type=int, default=0, help="Priority of the job")
    sub.add_argument("--max-retry", type=int, default=0, help="Max number of attempts (including first run)")

    # cancel
    sub = subparsers.add_parser("cancel", help="Cancel a job")
    sub.add_argument("job_id", type=int)
    
    # freeze
    sub = subparsers.add_parser("freeze", help="Freeze the environment")
    
    # scan
    sub = subparsers.add_parser("scan", help="Scan for vacant TPUs")
    sub.add_argument("--tpu-size", nargs="+", required=True)
    sub.add_argument("--region", nargs="+", required=True)

    # daemon
    sub = subparsers.add_parser("start", help="Run the scheduler daemon")
    sub.add_argument("--tick-interval", type=int, default=5, help="Seconds between main daemon ticks")
    sub.add_argument("--steal-wait", type=int, default=300, help="Seconds to wait before stealing (-1 to disable)")
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
            state_dir=args.state_dir,
            priority=args.priority,
            max_retry=args.max_retry,
        )
    elif args.action == "cancel":
        cancel_job(args.job_id, state_dir=args.state_dir)
    elif args.action == "scan":
        vacant = scan_target(args.tpu_size, args.region)
        thread_log(f"\n{len(vacant)} vacant TPU(s) found.", force_print=True)
    elif args.action == "freeze":
        freeze()
    elif args.action == "start":
        freeze()
        scheduler = Scheduler(
            alloc_max=args.alloc_max,
            alloc_sizes=args.alloc_sizes,
            alloc_regions=args.alloc_regions,
            alloc_workers=args.alloc_workers,
            init_workers=args.init_workers,
            steal_wait=args.steal_wait,
            steal_max=args.steal_max,
            tick_interval=args.tick_interval,
            state_dir=args.state_dir,
        )
        scheduler.run()

    elif args.action == "stop":
        stop_file = Path(args.state_dir) / "tpurm.stop"
        stop_file.touch()
        thread_log(f"Stop file created: {stop_file}", force_print=True)
    else:
        parser.print_help()
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
