#!/usr/bin/env bash

# Steps:
# 1. disable unattended-upgrades
# 2. apt install nfs-common (default: 20 retries)
# 3. mount NFS volumes
# 4. delete ~/.local

set -euo pipefail

# disable unattended upgrades
sudo systemctl stop unattended-upgrades.service || true
sudo systemctl disable unattended-upgrades.service || true
sudo systemctl stop apt-daily.service apt-daily-upgrade.service unattended-upgrades || true

PIDS=$(ps -ef | grep -i unattended | grep -v 'grep' | awk '{print $2}') || true
if [ -n "$PIDS" ]; then
    echo "[init.sh] killing unattended processes: $PIDS"
    echo "$PIDS" | xargs -r sudo kill -9 || true
fi

while sudo fuser /var/lib/dpkg/lock >/dev/null 2>&1; do
    echo '[init.sh] waiting for apt lock release...'
    sleep 5
done

# apt install nfs-common and python3.13
export DEBIAN_FRONTEND=noninteractive
APT_RETRIES=${APT_RETRIES:-2}
ret=1
for attempt in $(seq 1 $APT_RETRIES); do
    echo "[init.sh] apt init attempt $attempt/$APT_RETRIES"
    sudo dpkg --configure -a || true
    sudo apt-get -y -f install || true
    sudo add-apt-repository -y ppa:deadsnakes/ppa || true
    sudo apt-get -y update || true
    sudo apt-get -y install nfs-common python3.13 python3.13-venv python3.13-dev >/dev/null && ret=0 || ret=$?
    if [ "$ret" -eq 0 ]; then
        break
    fi
    echo '[init.sh] apt init failed, killing unattended processes and retrying...'
    ps -ef | grep -i unattended | grep -v 'grep' | awk '{print "sudo kill -9 " $2}' | sh || true
    sleep 5
done
if [ "$ret" -ne 0 ]; then
    echo '[init.sh] ERROR: apt initialization failed after retries' >&2
    exit 7
fi

# mount NFS (idempotent)
sudo mkdir -p /kmh-nfs-us-mount
mountpoint -q /kmh-nfs-us-mount || sudo mount -o vers=3 10.26.72.146:/kmh_nfs_us /kmh-nfs-us-mount
sudo chmod go+rw /kmh-nfs-us-mount
ls /kmh-nfs-us-mount

sudo mkdir -p /kmh-nfs-ssd-us-mount
mountpoint -q /kmh-nfs-ssd-us-mount || sudo mount -o vers=3 10.97.81.98:/kmh_nfs_ssd_us /kmh-nfs-ssd-us-mount
sudo chmod go+rw /kmh-nfs-ssd-us-mount
ls /kmh-nfs-ssd-us-mount

# clean up user local (avoids stale pip packages from TPU runtime image)
sudo rm -rf /home/$(whoami)/.local || true

# bootstrap pip for python3.13 (after cleanup so it's not deleted)
python3.13 -m ensurepip --upgrade
