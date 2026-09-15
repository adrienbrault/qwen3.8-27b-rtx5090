#!/usr/bin/env bash
# shm-sweep.sh — reclaim orphaned /dev/shm segments left by killed vLLM containers.
#
# WHY THIS EXISTS (2026-09-16). The sweep below used to live only inside launch-daily-v0280.sh, i.e. only on the
# DAILY BOOT path. But experiments kill the daily with `docker rm -f` and, per AGENTS.md §12, a chained run
# SKIPS the daily restore between units -- so across any chain the sweep never ran and orphans piled up.
#
# That is not theoretical. On 2026-09-15 the daily was killed at ~23:00 for R329 leaving a 16 GB
# vllm_offload_*.mmap behind; R329's restore was skipped (r331 was queued), so the sweep never fired; R329's
# `--ngram_ram` arm (32.6 GB n-gram table into a 60 GB box) then had only 40 GB available and was SIGKILLed by
# earlyoom at 23:16:00 with memory at 2.27%, leaving no traceback and no dmesg line. The arm looked like an
# engine bug for a day. It was 16 GB of stale shared memory.
#
# Upstream causes, both already documented in launch-daily-v0280.sh:
#   - vLLM OffloadingConnector's CPU staging mmap (/dev/shm/vllm_offload_*.mmap, ~4 GB each) survives `docker rm -f`.
#   - vLLM's multiprocessing shm_broadcast segments (/dev/shm/psm_*, 25-250 MB each) leak the same way.
#
# RULE: delete only ORPHANS (fuser reports no holder). Sweeping live segments would break a running engine.
set -uo pipefail
quiet="${1:-}"
now(){ echo "$(date -Is) [shm-sweep] $*"; }
avail(){ free -m | awk '/Mem:/{print $7}'; }
before_shm=$(df -m /dev/shm | awk 'NR==2{print $3}')
before_av=$(avail)

n=0
for pat in '/dev/shm/vllm_offload_*.mmap' '/dev/shm/psm_*'; do
  for f in $pat; do
    [ -e "$f" ] || continue
    sudo fuser -s "$f" 2>/dev/null || { sudo rm -f "$f" && n=$((n+1)); }
  done
done

after_shm=$(df -m /dev/shm | awk 'NR==2{print $3}')
after_av=$(avail)
if [ "$n" -gt 0 ]; then
  now "reclaimed $n orphan segment(s): /dev/shm ${before_shm}->${after_shm} MiB used, MemAvailable ${before_av}->${after_av} MiB"
  [ "$after_av" -lt 6000 ] && now "WARNING: MemAvailable still ${after_av} MiB — a large allocation may be killed by earlyoom (which SIGKILLs without a traceback or a dmesg line; check: journalctl -u earlyoom)"
else
  [ "$quiet" = quiet ] || now "no orphans (MemAvailable ${after_av} MiB)"
fi
exit 0
