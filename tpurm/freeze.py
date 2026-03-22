import subprocess
import sys

from .globals import REPO_ROOT

# Packages that are conda-specific or not needed on remote TPU VMs.
EXCLUDE = {
    "tpurm",
    "gmpy2",
    "Brotli",
    "munkres",
    "pybind11",
    "pybind11-global",
    "backports.zstd",
    "h2",
    "hpack",
    "hyperframe",
    "PySocks",
    "torchvision-extra-decoders",
    "pip",
    "setuptools",
}

def freeze():
    """Freeze current pip environment into requirements.lock."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=freeze"],
        capture_output=True, text=True, check=True,
    )

    included = []
    excluded = []
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split("==")[0]
        if name in EXCLUDE:
            excluded.append(line)
        else:
            included.append(line)

    lockfile = REPO_ROOT / "requirements.lock"
    lockfile.write_text("\n".join(included) + "\n")

    print(f"Wrote {len(included)} packages to {lockfile}")
    if excluded:
        print(f"Excluded {len(excluded)} packages: {', '.join(e.split('==')[0] for e in excluded)}")
