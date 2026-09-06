#!/usr/bin/env bash
# R205c (2026-09-06): WHY does the MTP image still get 0 prefix-cache hits live? R205 (mtpcache image, 0152/0154/0155/0156) measured
# prefix_hits=0 over 4.3M queries and every revisit at a cold-prefill ttft (7.8 s) while the DFlash picks image on the same probe
# hits 52,768 of 53,453. The REAL Scheduler driven offline (r205/r205_sched_repro.py, same image, hybrid 3 Mamba + 1 full-attn layout,
# block=hash=1552, chunk 8192, live prompt shape 53,468, MTP spec config method=mtp ns3) returns 51,216 hits in every variant, so the
# difference is engine-side and only observable live. This unit boots the mtpcache image on :8029 (same route as R205) with a LOGGING
# OVERLAY bind-mounted over kv_cache_manager.py / kv_cache_coordinator.py (no rebuild; "R205DBG" lines: get_computed_blocks input +
# result, find_longest_cache_hit per-group hit lengths, cache_blocks per-manager tokens-to-cache and cached-block counts), runs
# warm_equal (1×6K, twice) and warm-revisit (32K), and keeps the engine log. Expected under a working cache at ctx 6009: hit ≈ 4656−1552.
#   unit: sudo systemd-run --unit=r205c-mtp-hit-debug --collect -p User=adrienbrault -p RuntimeMaxSec=43200 -p TimeoutStopSec=900 \
#         -E GPU_QUEUE_NAME=r205c-mtp-hit-debug bash /srv/qwen5090/r205c-mtp-hit-debug.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-06-r205c-mtp-hit-debug; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
U=http://127.0.0.1:8029; CAND=/srv/qwen5090/launch-daily.sh; L2=/srv/qwen5090/eval-l2; PR=/srv/qwen5090/probes
OV=/srv/qwen5090/r205/overlay; SITE=/usr/local/lib/python3.12/dist-packages
DAILY=$(sed -nE 's/^DAILY_IMG=([^ ]+).*/\1/p' "$CAND" | head -1); IMG=$DAILY-mtppcie-mtpcache
PIN=13980000000; PIN_FALLBACK=13500000000
[ -n "$DAILY" ] || { log "ABORT: cannot read DAILY_IMG"; exit 3; }
sudo docker image inspect "$IMG" >/dev/null 2>&1 || { log "ABORT: image $IMG missing"; exit 3; }
grep -q "R183 EXP-only passthrough" "$CAND" || { log "ABORT: launch-daily.sh lacks the R183 EXP passthrough (EXTRA_MOUNT_APPEND)"; exit 3; }
for t in warm-revisit.py warm_equal.py reask.py; do [ -f "$PR/$t" ] || { log "ABORT: $PR/$t missing"; exit 3; }; done

# ---- overlay: extract the two modules from the image and add the R205DBG log lines (CPU, no lock needed) ----
mkdir -p "$OV"
for f in kv_cache_manager kv_cache_coordinator; do sudo docker run --rm --entrypoint cat "$IMG" "$SITE/vllm/v1/core/$f.py" > "$OV/$f.orig.py" || { log "ABORT: cannot read $f.py from $IMG"; exit 3; }; done
python3 - "$OV" <<'PY' || { log "ABORT: overlay patch failed"; exit 3; }
import sys, pathlib
ov = pathlib.Path(sys.argv[1])
def sub(text, old, new, name):
    n = text.count(old)
    assert n == 1, f"{name}: expected 1 occurrence, found {n}: {old[:60]!r}"
    return text.replace(old, new)
m = (ov / "kv_cache_manager.orig.py").read_text()
m = sub(m, """        if not self.prefix_cache_lookup_enabled(request):
            return self.empty_kv_cache_blocks, 0, 0
""", """        if not self.prefix_cache_lookup_enabled(request):
            logger.info("R205DBG gcb SKIP req=%s enable_caching=%s skip_read=%s", request.request_id, self.enable_caching, request.skip_reading_prefix_cache)
            return self.empty_kv_cache_blocks, 0, 0
""", "manager skip")
m = sub(m, """        computed_blocks, num_new_computed_tokens, num_uncached = (
            self.coordinator.find_longest_cache_hit(
                request.block_hashes, max_cache_hit_length
            )
        )
""", """        computed_blocks, num_new_computed_tokens, num_uncached = (
            self.coordinator.find_longest_cache_hit(
                request.block_hashes, max_cache_hit_length
            )
        )
        logger.info("R205DBG gcb req=%s num_tokens=%d prompt=%d hashes=%d max=%d hit=%d uncached=%d blocks_per_group=%s coordinator=%s",
                    request.request_id, request.num_tokens, request.num_prompt_tokens, len(request.block_hashes), max_cache_hit_length,
                    num_new_computed_tokens, num_uncached, [len(b) for b in computed_blocks], type(self.coordinator).__name__)
""", "manager result")
(ov / "kv_cache_manager.py").write_text(m)
c = (ov / "kv_cache_coordinator.orig.py").read_text()
c = sub(c, """        num_uncached_common_prefix_tokens = longest_hit_length - hit_length
        cache_hit_blocks = tuple(
            blocks if blocks is not None else [] for blocks in hit_blocks_by_group
        )
        return cache_hit_blocks, hit_length, num_uncached_common_prefix_tokens
""", """        num_uncached_common_prefix_tokens = longest_hit_length - hit_length
        cache_hit_blocks = tuple(
            blocks if blocks is not None else [] for blocks in hit_blocks_by_group
        )
        logger.info("R205DBG flch nhash=%d max=%d hit=%d longest=%d by_group=%s eagle_groups=%s partial=%s managers=%s",
                    len(block_hashes), max_cache_hit_length, hit_length, longest_hit_length, hit_length_by_group,
                    sorted(self.eagle_group_ids), getattr(self, "enable_partial_hash_hits", None),
                    [(type(mg).__name__, mg.block_size, getattr(mg, "use_eagle", None)) for mg in self.single_type_managers])
        return cache_hit_blocks, hit_length, num_uncached_common_prefix_tokens
""", "coordinator flch")
c = sub(c, """    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        cached_num_computed_tokens = self._align_cacheable(num_computed_tokens)
        for manager in self.single_type_managers:
            num_tokens_to_cache = cached_num_computed_tokens
""", """    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        cached_num_computed_tokens = self._align_cacheable(num_computed_tokens)
        _dbg = []
        for manager in self.single_type_managers:
            num_tokens_to_cache = cached_num_computed_tokens
""", "coordinator cache_blocks head")
c = sub(c, """            manager.cache_blocks(
                request,
                num_tokens_to_cache,
                retention_interval=self.retention_interval,
            )

    def find_longest_cache_hit(
""", """            _dbg.append(num_tokens_to_cache)
            manager.cache_blocks(
                request,
                num_tokens_to_cache,
                retention_interval=self.retention_interval,
            )
        _last = self.__dict__.setdefault("_r205_last", {})
        _cnt = [mg.num_cached_block.get(request.request_id) for mg in self.single_type_managers]
        _sig = (cached_num_computed_tokens, tuple(_cnt))
        if _last.get(request.request_id) != _sig or num_computed_tokens <= request.num_prompt_tokens:
            _last[request.request_id] = _sig
            logger.info("R205DBG cache_blocks req=%s num_computed=%d prompt=%d hashes=%d aligned=%d reprefillable=%d per_manager_tokens=%s cached_blocks=%s",
                        request.request_id, num_computed_tokens, request.num_prompt_tokens, len(request.block_hashes), cached_num_computed_tokens,
                        self.num_reprefillable_tokens, _dbg, _cnt)

    def find_longest_cache_hit(
""", "coordinator cache_blocks tail")
(ov / "kv_cache_coordinator.py").write_text(c)
print("overlay written")
PY
python3 -m py_compile "$OV/kv_cache_manager.py" "$OV/kv_cache_coordinator.py" || { log "ABORT: overlay does not compile"; exit 3; }
log "overlay ready: $(grep -c R205DBG "$OV/kv_cache_manager.py") + $(grep -c R205DBG "$OV/kv_cache_coordinator.py") log sites"
XM="-v $OV/kv_cache_manager.py:$SITE/vllm/v1/core/kv_cache_manager.py:ro -v $OV/kv_cache_coordinator.py:$SITE/vllm/v1/core/kv_cache_coordinator.py:ro"

. /srv/qwen5090/lib/gpu-queue.sh
HAVE_LOCK=0; exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R205c $1 ==="; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then finish ABORTED; else log "no lock held: engines left alone, exiting"; fi; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
HAVE_LOCK=1
log "=== R205c start (lock held): $IMG + R205DBG overlay, MTP ns3, same route as R205 ==="
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; finish ABORTED; exit 1; }
wipe_l2(){ sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; }
ELOG(){ sudo docker logs vllm-exp 2>&1; }
teardown; wipe_l2
boot_once(){ local tag=$1 kv=$2 rc
  env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=16 KV_BYTES=$kv PCIE_IPC=1 BSS=1 CAND_IMG=$IMG SPEC_METHOD=mtp SPEC_NS=3 EXTRA_MOUNT_APPEND="$XM" EXTRA_ENV_APPEND="-e VLLM_TRITON_FORCE_FIRST_CONFIG=1 -e VLLM_SM12X_PCIE_IPC_MTP=1" bash $CAND > "$R/boot-$tag-$kv.log" 2>&1; rc=$?
  if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then
    ELOG > "$R/engine-boot-$tag.log"
    log "[$tag] BOOT OK pin=$kv pool=$(grep -aoE 'Pool [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9) block=$(grep -aoE 'Setting attention block size to [0-9]+' "$R/engine-boot-$tag.log" | head -1 | tr -dc 0-9) overlay_mounted=$(sudo docker inspect vllm-exp --format '{{range .Mounts}}{{.Destination}} {{end}}' | grep -c kv_cache_coordinator)"
    return 0; fi
  log "[$tag] boot pin=$kv FAILED rc=$rc: $(grep -aE 'FAILED' "$R/boot-$tag-$kv.log" | tail -1 | cut -c1-220)"
  ELOG 2>/dev/null | grep -aiE "error|exception|R205DBG" | head -8 | cut -c1-220 | sed "s/^/[$tag boot-err] /" | tee -a "$R/audit.log"
  ELOG > "$R/engine-bootfail-$tag.log" 2>/dev/null; teardown; return 1; }
if boot_once DBG $PIN; then :; elif boot_once DBGb $PIN_FALLBACK; then :; else log "BOOT FAILED at both pins"; finish ABORTED; exit 1; fi
log "[warn] $(grep -aoE "could be identified as the draft model's[^\"]{0,200}" "$R/engine-boot-DBG"*.log | head -1 | cut -c1-260)"
snap(){ curl -s -m 5 $U/metrics | awk -v t="$1" '/^vllm:prefix_cache_(queries|hits)_total/ {split($1,a,"{"); q[a[1]]+=$NF} END {printf "[metrics %s] prefix_queries=%d prefix_hits=%d\n", t, q["vllm:prefix_cache_queries_total"], q["vllm:prefix_cache_hits_total"]}' | tee -a "$R/audit.log"; }
sleep 10; snap boot
python3 $PR/warm_equal.py --url $U --model qwen3.8-27b --n 1 --ctx 6000 --max-tokens 16 > "$R/warm-equal-6k.log" 2>&1
log "[warm-equal 6K] $(tail -1 "$R/warm-equal-6k.log") $(grep -a RESULT "$R/warm-equal-6k.log" | cut -c1-300)"
snap after-6k
python3 $PR/warm-revisit.py --url $U --model qwen3.8-27b --ctx 32000 > "$R/revisit-32k.log" 2>&1
log "[revisit 32K] $(grep -a RESULT "$R/revisit-32k.log" | cut -c1-330)"
snap after-32k
# contrast matrix (needle_depth's 2nd re-ask HIT 129,536/131,245 with chat + long answer + 15 s gap; warm-revisit = completions + 8 tokens + no gap never hits)
reask(){ local name=$1; shift 1
  python3 $PR/reask.py --url $U --model qwen3.8-27b --ctx 6000 "$@" > "$R/reask-$name.log" 2>&1
  log "[reask $name] $(grep -a RESULT "$R/reask-$name.log" | cut -c1-360)"; }
reask compl-mt8-gap0   --api completions --max-tokens 8   --gap 0  --sends 3 --seed a
reask compl-mt128-gap0 --api completions --max-tokens 128 --gap 0  --sends 2 --seed b
reask compl-mt8-gap15  --api completions --max-tokens 8   --gap 15 --sends 2 --seed c
reask chat-mt128-gap15 --api chat        --max-tokens 128 --gap 15 --sends 2 --seed d
reask chat-mt8-gap0    --api chat        --max-tokens 8   --gap 0  --sends 2 --seed e
snap after-reasks
sleep 3; ELOG > "$R/engine-full.log"
grep -a R205DBG "$R/engine-full.log" | sed -E 's/^.*R205DBG/R205DBG/' > "$R/r205dbg.txt"
log "[dbg] $(wc -l < "$R/r205dbg.txt") R205DBG lines; gcb=$(grep -c '^R205DBG gcb ' "$R/r205dbg.txt") skip=$(grep -c 'gcb SKIP' "$R/r205dbg.txt") flch=$(grep -c 'R205DBG flch' "$R/r205dbg.txt") cache_blocks=$(grep -c 'R205DBG cache_blocks' "$R/r205dbg.txt")"
grep -a '^R205DBG gcb\|^R205DBG flch' "$R/r205dbg.txt" | cut -c1-400 | sed 's/^/[dbg] /' | tee -a "$R/audit.log"
grep -a '^R205DBG cache_blocks' "$R/r205dbg.txt" | awk '{print} NR>=40{exit}' | cut -c1-300 | sed 's/^/[dbg] /' | tee -a "$R/audit.log"
log "[engine error-lines] $(ELOG | grep -ac 'illegal memory\|CUDA error\|Traceback\|OutOfMemoryError')"
finish DONE
