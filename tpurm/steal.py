from concurrent.futures import ThreadPoolExecutor, as_completed

from .tpu import TPU_CONFIGS, zone_to_region, name_to_tpu
from .util_log import LogContext
from .util_ssh import check_vacancy
from .util_gcloud import gcloud_list, gcloud_describe

def scan_target(tpu_sizes: list[str], regions: list[str], *, log_ctx: LogContext) -> list[tuple[str, str]]:
    """Returns the list of vacant TPUs matching the requested size(s) and region(s)."""
    zones = []
    family_to_chips: dict[str, set[int]] = {}
    for tpu_size in tpu_sizes:
        family, n_chips = tpu_size.split("-")
        cfg = TPU_CONFIGS[family]
        family_to_chips.setdefault(family, set()).add(int(n_chips))
        zones.extend([z for z in cfg["allowed_zones"] if zone_to_region(z) in regions])

    candidates = []  # (name, zone)
    for zone in sorted(set(zones)):
        vms = gcloud_list(zone, log_ctx=log_ctx)
        for vm in vms:
            name = vm.get("name", "").rsplit("/", 1)[-1]
            tpu = name_to_tpu(name, zone)
            if tpu is None:
                log_ctx.log(f"Could not parse TPU name: {name}, skipping it.")
                continue
            family, n_chips = tpu.size.split("-")
            chips = int(n_chips)
            if (
                family not in family_to_chips 
                or not any(req <= chips <= 2 * req for req in family_to_chips[family])
            ):
                continue

            status = vm.get("state", "")
            if status != "READY":
                continue
            candidates.append((name, zone))

    if not candidates:
        log_ctx.log("No READY VMs found matching the requested size(s).")
        return []

    log_ctx.log(f"Checking vacancy for {len(candidates)} VMs...")

    def check_one(name, zone):
        info = gcloud_describe(name, zone, log_ctx=log_ctx)
        if info is None:
            log_ctx.log(f"Skipping {name}: gcloud_describe failed.")
            return None
        if (
            info["state"] != "READY" or
            info["health"] != "HEALTHY"
        ):
            log_ctx.log(f"Skipping {name}: state={info['state']}, health={info['health']}.")
            return False, f"{info["state"] or "N/A"}/{info["health"] or "N/A"}"
        return check_vacancy(name, zone, log_ctx=log_ctx)

    results = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(check_one, name, zone): (name, zone)
            for name, zone in candidates
        }
        for fut in as_completed(futures):
            name, zone = futures[fut]
            info = fut.result()
            results[(name, zone)] = info

    # Print summary table
    log_ctx.log(f"{'VM Name':<55} {'Zone':<20} {'Vacant':<10} {'Load'}")
    log_ctx.log("-" * 100)
    vacant_vms = []
    for name, zone in candidates:
        info = results[(name, zone)]
        if info is None:
            status_str = "Dead"
            continue
        if info[0]:
            status_str = "Yes"
            vacant_vms.append((name, zone))
        else:
            status_str = "No"
        log_ctx.log(f"{name:<55} {zone:<20} {status_str:<10} {info[1]}")
    
    return vacant_vms
