# Source at the top of every GPU-exclusive experiment script, BEFORE the flock (2026-09-03, user: "control yourself whether
# or not to restore the daily, so that between experiments we don't have to wait for daily up/down").
# Registers this run in /srv/qwen5090/gpu-queue/<name> (content: PID). daily-restore-retry.sh skips the restore while
# another LIVE registration exists (a unit queued on the flock), so a chain of experiments pays one daily down/up, not one
# per unit; the last unit restores. Dead-PID markers are ignored and removed. FORCE_RESTORE=1 overrides.
#   . /srv/qwen5090/lib/gpu-queue.sh            # name = script basename; or GPU_QUEUE_NAME=... before sourcing
GPU_QUEUE_DIR=/srv/qwen5090/gpu-queue; mkdir -p "$GPU_QUEUE_DIR"
GPU_QUEUE_MARK="$GPU_QUEUE_DIR/${GPU_QUEUE_NAME:-$(basename "$0" .sh)}"
export GPU_QUEUE_SELF=$$; echo "$$" > "$GPU_QUEUE_MARK"
trap 'rm -f "$GPU_QUEUE_MARK"' EXIT
gpu_queue_others(){ local m pid; for m in "$GPU_QUEUE_DIR"/*; do [ -e "$m" ] || continue; pid=$(cat "$m" 2>/dev/null)
  [ "$pid" = "$GPU_QUEUE_SELF" ] && continue; if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then echo "$(basename "$m")(pid $pid)"; else rm -f "$m"; fi; done; }
