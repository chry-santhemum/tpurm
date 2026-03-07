"""Scan for vacant TPU VMs."""

from concurrent.futures import ThreadPoolExecutor, as_completed

from .common import (
    TPU_CONFIGS,
    thread_log, zone_to_region, name_to_tpu,
    check_vacancy, list_tpus, set_thread_vars, _thread_local
)


def get_zone_from_name(name: str) -> str:
    """Find which zone a TPU lives in by parsing its family and scanning allowed zones."""
    import re
    m = re.match(r"^kmh-tpuvm-(v4|v5e|v5p|v6e)-", name)
    if not m:
        raise ValueError(f"Cannot parse TPU family from name: {name}")
    family = m.group(1)
    allowed_zones = TPU_CONFIGS[family]["allowed_zones"]
    for zone in allowed_zones:
        for vm in list_tpus(zone):
            vm_name = vm.get("name", "").rsplit("/", 1)[-1]
            if vm_name == name:
                return zone
    raise ValueError(f"TPU {name} not found in any zone for family {family}: {allowed_zones}")


def scan_target(tpu_sizes: list[str], regions: list[str]) -> list[tuple[str, str]]:
    zones = []
    family_to_chips: dict[str, set[int]] = {}
    for tpu_size in tpu_sizes:
        family, n_chips = tpu_size.split("-")
        cfg = TPU_CONFIGS[family]
        family_to_chips.setdefault(family, set()).add(int(n_chips))
        zones.extend([z for z in cfg["allowed_zones"] if zone_to_region(z) in regions])

    candidates = []  # (name, zone)
    for zone in sorted(set(zones)):
        vms = list_tpus(zone)
        for vm in vms:
            name = vm.get("name", "").rsplit("/", 1)[-1]
            tpu = name_to_tpu(name, zone)
            if tpu is None:
                thread_log(f"[steal.py] Could not parse TPU name: {name}. Continuing.")
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
        thread_log("No READY VMs found matching the requested size(s).")
        return []

    thread_log(f"Checking vacancy for {len(candidates)} VMs...")
    log_file = getattr(_thread_local, 'log_file', None)

    def _check(name, zone):
        with set_thread_vars(log_file=log_file):
            return check_vacancy(name, zone)

    results = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(_check, name, zone): (name, zone)
            for name, zone in candidates
        }
        for fut in as_completed(futures):
            name, zone = futures[fut]
            info = fut.result()
            results[(name, zone)] = info

    # Print summary table
    thread_log("")
    thread_log(f"{'VM Name':<55} {'Zone':<20} {'Vacant':<10} {'Load'}")
    thread_log("-" * 100)
    vacant_vms = []
    for name, zone in candidates:
        info = results[(name, zone)]
        if info["vacant"] is None:
            status_str = "???"
        elif info["vacant"]:
            status_str = "YES"
        else:
            status_str = "no"
        load = info["load"].split()[0] if info["load"] else "?"
        thread_log(f"{name:<55} {zone:<20} {status_str:<10} {load}")
        if info["vacant"]:
            vacant_vms.append((name, zone))
    
    return vacant_vms
