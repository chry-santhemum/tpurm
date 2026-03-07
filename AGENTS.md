# TPURM Scheduler Notes

`state.json` is the single source of truth across processes. Every mutation goes through `FileState.transact()` so the scheduler and CLI commands coordinate by file lock, not by in-memory objects.

The scheduler intentionally treats TPU and job lifecycle as separate state machines. TPU tracking (`need_init`/`initializing`/`free`/`busy`) is updated by sync/init/alloc paths, while job status (`queued`/`running`/terminal states) is updated by matching, polling, finalize, and cancel paths.

On startup, the daemon does not try to reconstruct per-thread launch context. It syncs tracked TPUs, resets stuck `initializing` TPUs to `need_init`, backfills missing `log_dir` for `running` jobs from `stage_dir`, and immediately polls once to reconcile already-finished runs.

Each main tick is ordered deliberately: stop gate -> poll running jobs -> sync TPU status -> update alloc sleep signal -> match queued jobs -> brief pre-launch recheck -> launch outside lock -> optionally run steal logic -> sleep.

Matching and launching are decoupled on purpose. Matching marks TPU `busy`, sets job `running`, and increments `attempt` under lock; actual remote launch happens outside the lock so network latency does not block state progress.

Before each launch, the scheduler rechecks that the job is still `running` and still assigned to the same TPU. This is the intended protection against cross-process cancellation races.

`poll_jobs()` only tracks `running` jobs. Cancelled jobs are handled by `cancel_job()`/`finalize_job()` state transitions and are not part of ongoing polling.

`finalize_job()` is the only place that decides success/failure/requeue after a launch attempt. It also releases TPU occupancy and clears `assigned_tpu` on terminal or cancellation paths so jobs cannot be repeatedly re-finalized.

`max_retry` is treated as max attempts (current behavior), so retry decisions are based on `attempt` count already incremented at match time.

Stealing is a reservation flow: a queued job may carry a pre-assigned TPU before launch. Those reserved stolen TPUs still consume steal budget checks when transitioning to running.

Allocator workers only create owned TPUs (`need_init`). Init workers are the only writers that claim `need_init` TPUs and move them through initialization.

`sync_tracked_tpus()` is the reconciliation boundary with cloud reality. It updates tracked TPU status/worker count, untracks missing TPUs, and keeps running-job TPUs from being treated as idle while jobs are active.

Cancellation is intentionally side-effectful outside the scheduler loop: it marks job state first, then kills local/remote processes, then frees tracked TPU state if still present.
