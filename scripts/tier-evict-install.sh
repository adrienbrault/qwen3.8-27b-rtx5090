#!/usr/bin/env bash
# Installs tier-evict.sh as a 2-minute systemd timer on flan (run ON flan after shipping tier-evict.sh
# to /srv/qwen5090/). Also remounts the tier with strictatime,lazytime so eviction is true LRU
# (lazytime keeps atime updates in memory; setup-gen5-tier.sh carries the fstab line).
set -euo pipefail
MNT=/srv/qwen5090/native-l2
sudo mount -o remount,strictatime,lazytime "$MNT" && grep " $MNT " /proc/mounts
sudo sed -i 's#^\(LABEL=gen5-tier /srv/qwen5090/native-l2 ext4\) [^ ]*#\1 strictatime,lazytime#' /etc/fstab && grep gen5-tier /etc/fstab
sudo tee /etc/systemd/system/tier-evict.service >/dev/null <<'U'
[Unit]
Description=flan KV disk-tier LRU evictor (vLLM fs tier has none)
[Service]
Type=oneshot
User=adrienbrault
ExecStart=/usr/bin/bash /srv/qwen5090/tier-evict.sh
U
sudo tee /etc/systemd/system/tier-evict.timer >/dev/null <<'U'
[Unit]
Description=flan KV disk-tier evictor, every 2 min
[Timer]
OnBootSec=2min
OnUnitActiveSec=2min
AccuracySec=15s
[Install]
WantedBy=timers.target
U
sudo systemctl daemon-reload && sudo systemctl enable --now tier-evict.timer && systemctl list-timers tier-evict.timer --no-legend
