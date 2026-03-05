import argparse
import time
from pathlib import Path

from .common import thread_log
from .scheduler import FILE_STATE_DIR, Allocator, Scheduler, cancel_job, submit_job
from .steal import scan_target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TPU Job Scheduler")
    parser.add_argument(
        "--state-dir", default=FILE_STATE_DIR,
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
    sub.add_argument("--max-retry", type=int, default=3, help="Max number of retries on failure")

    # cancel
    sub = subparsers.add_parser("cancel", help="Cancel a job")
    sub.add_argument("job_id", type=int)

    # scan
    sub = subparsers.add_parser("scan", help="Scan for vacant TPUs")
    sub.add_argument("--tpu-size", nargs="+", required=True)
    sub.add_argument("--region", nargs="+", required=True)

    # daemon
    sub = subparsers.add_parser("start", help="Run the scheduler daemon (includes allocator)")
    sub.add_argument("--alloc-tick-interval", type=int, default=15, help="Seconds between allocator ticks")
    sub.add_argument("--sched-tick-interval", type=int, default=30, help="Seconds between scheduler ticks")
    sub.add_argument("--steal-wait", type=int, default=300, help="Seconds to wait before stealing (-1 to disable)")
    sub.add_argument("--max-steal", type=int, default=1, help="Max jobs on stolen TPUs (-1 for unlimited)")
    sub.add_argument("--target-num", type=int, default=4, help="Target number of owned TPUs (0 to disable allocation)")
    sub.add_argument("--sizes", nargs="+", default=["v6e-64"], help="TPU sizes to allocate")
    sub.add_argument("--regions", nargs="+", default=None, help="Restrict allocation to these regions")
    sub.add_argument("--max-alloc-workers", type=int, default=4, help="Parallel allocation workers")
    sub.add_argument("--max-init-workers", type=int, default=4, help="Parallel initialization workers")

    # stop
    subparsers.add_parser("stop", help="Gracefully stop the daemon")

    args = parser.parse_args(argv)

    if args.action == "submit":
        assert args.command or args.command_path, "Provide --command or --command-path"
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
    elif args.action == "start":
        allocator = Allocator(
            target_num=args.target_num,
            sizes=args.sizes or ["v6e-64", "v5p-64"],
            regions=args.regions,
            max_alloc_workers=args.max_alloc_workers,
            max_init_workers=args.max_init_workers,
            tick_interval=args.alloc_tick_interval,
            state_dir=args.state_dir,
        )
        allocator.run()

        scheduler = Scheduler(
            steal_wait=args.steal_wait,
            max_steal=args.max_steal,
            tick_interval=args.sched_tick_interval,
            state_dir=args.state_dir,
        )
        try:
            scheduler.run()
        finally:
            allocator.stop()
            allocator.join()

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
