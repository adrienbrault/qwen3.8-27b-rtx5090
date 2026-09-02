#!/usr/bin/env bash
# earlyoom on flan (2026-09-03, after the 09-02 OOM → thrash → hung_task panic reboot): kill the largest
# offender at 5% MemAvailable (SIGKILL at 2.5%) BEFORE the kernel thrashes into the 300 s hung-task panic.
# Prefers interpreter/test processes (what campaign containers run), avoids the engine, docker, k3s, ssh.
# The vLLM containers additionally run with --oom-score-adj -800 (launch-daily-v0280.sh), which both
# earlyoom (sorts by oom_score) and the kernel OOM killer honour. Idempotent:
#   ssh flan 'sudo bash -s' < flan/setup-earlyoom.sh
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
dpkg -s earlyoom >/dev/null 2>&1 || apt-get install -y -q earlyoom >/dev/null
cat > /etc/default/earlyoom <<'CFG'
# managed by kubernetes-home/flan/setup-earlyoom.sh
EARLYOOM_ARGS="-m 5 -s 50 -r 300 --prefer '^(python3?(\.[0-9]+)?|pytest|node|java|pip3?|cc1plus|ld|make)$' --avoid '^(VLLM|vllm|dockerd|containerd|k3s|sshd|systemd|nvidia|netdata|zfs|txg)'"
CFG
systemctl enable --now earlyoom >/dev/null 2>&1 || true
systemctl restart earlyoom
sleep 1; systemctl is-active earlyoom; journalctl -u earlyoom --no-pager -n 3 -o cat | cut -c1-160
