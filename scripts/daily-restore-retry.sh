#!/usr/bin/env bash
# Restore the daily with one retry (heavy-TP2 transient "CUDA error: invalid argument" at warmup).
# Vendored 2026-09-02 (was on flan only since R159 — repo-first slip). env -i: no experiment knobs leak.
export PATH="$HOME/.local/bin:$PATH"
# Queue-aware (2026-09-03, user): experiment scripts that source lib/gpu-queue.sh register in /srv/qwen5090/gpu-queue/;
# if another live registration exists (a unit waiting on the GPU flock), skip the restore — the chain's last unit restores.
# FORCE_RESTORE=1 restores regardless. Dead-PID markers are ignored and removed.
if [ "${FORCE_RESTORE:-0}" != 1 ]; then
  for m in /srv/qwen5090/gpu-queue/*; do [ -e "$m" ] || continue; pid=$(cat "$m" 2>/dev/null); [ "$pid" = "${GPU_QUEUE_SELF:-}" ] && continue
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then echo "DAILY RESTORE SKIPPED: $(basename "$m") (pid $pid) is queued for the GPUs next"; exit 0; else rm -f "$m"; fi
  done
fi
for att in 1 2; do
  for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "\$1>1024{c++} END{print c+0}"); [ "$busy" = 0 ] && break; sleep 5; done
  [ $att = 1 ] && sleep 60 || sleep 300   # 2026-09-02: a 300 s settle cleared the repeated TP1-warmup "invalid argument" + TP0 hang (post-reboot); R162: 60 s boots first try on a healthy box
  if env -i HOME="$HOME" USER="$USER" PATH="$PATH" bash /srv/qwen5090/launch-daily.sh > /tmp/daily-restore-$att.log 2>&1; then echo "DAILY RESTORED attempt $att"; exit 0; fi
  echo "attempt $att FAILED: $(grep -aE "FAILED" /tmp/daily-restore-$att.log | tail -1 | cut -c1-120)"; sudo docker logs vllm-27b > /tmp/daily-fail-$att-$(date +%H%M%S).log 2>&1; sudo docker rm -f vllm-27b >/dev/null 2>&1
done; echo "DAILY RESTORE FAILED TWICE"; exit 1
