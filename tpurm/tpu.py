import re
from typing import Any, Literal
from dataclasses import dataclass, field

from .globals import ENV_VARS

# TPU configuration
# To add a new config: add service accounts, bucket, bucket key
TPU_CONFIGS: dict[str, dict[str, Any]] = {
    "v4": {
        "allowed_zones": ["us-central2-b"],
        "runtime_version": "tpu-ubuntu2204-base",
        "accelerator_type": lambda size: size,
    },
    "v5e": {
        "allowed_zones": ["us-west4-a", "us-central1-a"],
        "runtime_version": "v2-alpha-tpuv5-lite",
        "accelerator_type": lambda size: f"v5litepod-{size.split('-')[1]}",
    },
    "v5p": {
        "allowed_zones": ["us-central1-a", "us-east5-a"],
        "runtime_version": "v2-alpha-tpuv5",
        "accelerator_type": lambda size: size,
    },
    "v6e": {
        "allowed_zones": ["us-central1-b", "us-east5-b", "asia-northeast1-b"],
        "runtime_version": "v2-alpha-tpuv6e",
        "accelerator_type": lambda size: size,
    },
}

REGION_SERVICE_ACCOUNTS: dict[str, str] = {    # type: ignore
    "us-west4": ENV_VARS["REGION_SERVICE_ACCOUNTS_US_WEST4"],
    "us-east5": ENV_VARS["REGION_SERVICE_ACCOUNTS_US_EAST5"],
    "us-central1": ENV_VARS["REGION_SERVICE_ACCOUNTS_US_CENTRAL1"],
    "us-central2": ENV_VARS["REGION_SERVICE_ACCOUNTS_US_CENTRAL2"],
    "asia-northeast1": ENV_VARS["REGION_SERVICE_ACCOUNTS_ASIA_NORTHEAST1"],
}

REGION_BUCKETS = {
    "us-west4": "gs://kmh-gcp-us-west4",
    "us-east5": "gs://kmh-gcp-us-east5",
    "us-central1": "gs://kmh-gcp-us-central1",
    "us-central2": "gs://kmh-gcp-us-central2",
    "asia-northeast1": "gs://kmh-gcp-asia-northeast1-b",
}

AllocMode = Literal["spot", "preemptible", "persistent"]

@dataclass
class TPU:
    """TPU properties known at creation."""
    size: str         # e.g. "v5p-64"
    mode: AllocMode   # e.g. "spot"
    owner: str        # e.g. "atticusw"
    id: str           # e.g. "260817"
    zone: str         # e.g. "us-central2-b"
    name: str = field(init=False)
    num_workers: int|None = None

    def __post_init__(self):
        self.name = f"kmh-tpuvm-{self.size}-{self.mode}-{self.owner}-{self.id}"
        self.family = size_to_family(self.size)
        self.region = zone_to_region(self.zone)
        self.wheelhouse_tag = self.family if self.family in ("v5p", "v6e") else ""
        self.config = TPU_CONFIGS[self.family]
        self.service_account: str = REGION_SERVICE_ACCOUNTS[self.region]
        self.bucket = REGION_BUCKETS[self.region]

        allowed_zones = self.config["allowed_zones"]
        if self.zone not in self.config["allowed_zones"]:
            raise ValueError(f"{self.family} support zones: {allowed_zones}. Requested: {self.zone}")

def zone_to_region(zone: str) -> str:
    region, part = zone.rsplit("-", 1)
    assert len(part) == 1 and part.islower(), f"Invalid zone format: {zone}"
    return region

def size_to_family(tpu_size: str) -> str:
    prefix, _ = tpu_size.split("-")
    if prefix in ("v4", "v5e", "v5p", "v6e"):
        return prefix
    raise ValueError(f"Invalid TPU size: {tpu_size}. Expected v4-*, v5e-*, v5p-*, or v6e-*.")

def name_to_tpu(name: str, zone: str) -> TPU | None:
    """
    Parse a VM name like `kmh-tpuvm-v6e-64-spot-atticusw-12345` into a TPU.
    The only requirement is the `kmh-tpuvm-{family}-{chips}` prefix.
    Returns None if cannot be parsed.
    Mode defaults to `spot` when it can't be detected.
    The constructed TPU's `.name` is overridden with the actual VM name.
    """
    m = re.match(r"^kmh-tpuvm-(v4|v5e|v5p|v6e)-(\d+)(?:-(.*))?$", name)
    if not m:
        return None
    family, chips, remainder = m.groups()
    size = f"{family}-{chips}"
    remainder = remainder or ""

    # Try to strip a known mode from the front
    mode = "spot"
    for candidate in ("spot", "preemptible", "persistent"):
        if remainder == candidate:
            mode = candidate
            remainder = ""
            break
        if remainder.startswith(candidate + "-"):
            mode = candidate
            remainder = remainder[len(candidate) + 1:]
            break

    # Best-effort owner from what's left, rest is id (this is purely cosmetic)
    parts = remainder.split("-", 1) if remainder else []
    owner = parts[0] if parts else "unknown"
    tpu_id = parts[1] if len(parts) > 1 else ""

    tpu = TPU(size, mode, owner, tpu_id, zone)
    tpu.name = name  # override with actual VM name
    return tpu
