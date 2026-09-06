#!/usr/bin/env bash
# R207 promotion gate (2026-09-06, user "Alright let's switch to mtp"; sheet flan/r206-DECISION.md): the first boot of the MTP ns3 route ON
# THE SERVING PORT, gated the way R182/R197 were. Every MTP number so far is from :8029 (EXP=1, MIN_FREE floor 384 MiB, EXP pool band);
# launch-daily.sh now serves MTP ns3 (image ...-mtppcie-mtpcache-eagleshift, block 1,472, band 1.28–1.34M, VLLM_SM12X_PCIE_IPC_MTP=1 +
# the 0148/0158 boot asserts, CPU-tier floor 550 blocks) and this unit exercises those edits with a rollback:
#   teardown → boot from launch-daily.sh at 13.98 GB (13.5 GB retry on ANY failure, R191) → assert the tier wipe + stamp 1472 → layout/pool
#   → decode fidelity vs the r173c bf16 dumps at ctx 0 and 30K (the one gate the MTP route has never run: launcher header line 13 forbids
#     flipping the speculative route without an r173c-style ruler; ns7 was RETRACTED on it, band 0.0051–0.0062 at 30K)
#   → kv_capacity short / 100K / five 100K → needle gate with EVICT=16 (the MTP pool is 1,309,368: the default 12 × 90K = 1.08M cannot evict
#     and the re-asks would be served from the pool and misread as tier hits) → decode_ss README rows → tool-eval 69×4 → error lines.
# On a boot failure at both pins the frozen DFlash ns7 launcher (launch-daily-r197-dflash-ns7-0906.sh) brings the daily back. The fidelity
# ruler does NOT auto-roll back (the route switch is the user's decision); a reading outside the band leads the report.
# The daily stays UP at the end. native-l2 is wiped by the launcher (block 1,552 → 1,472): every warm context is lost, needles are cold.
#   unit: sudo systemd-run --unit=r207-promote-mtp --collect -p User=adrienbrault -p RuntimeMaxSec=43200 -p TimeoutStopSec=900 \
#         -E GPU_QUEUE_NAME=r207-promote-mtp bash -c '. /srv/qwen5090/lib/gpu-queue.sh; exec bash /srv/qwen5090/r207-promote-mtp.sh'
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-06-r207-promote-mtp; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
U=http://127.0.0.1:8020; PR=/srv/qwen5090/probes; L2=/srv/qwen5090/native-l2
ROLLBACK=/srv/qwen5090/launch-daily-r197-dflash-ns7-0906.sh
FD=/srv/qwen5090/results/2026-08-23-fidelity; DREF=/srv/qwen5090/results/2026-09-04-r173c-bf16-decode
IMG=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616-pcieipc-bsshash-mtppcie-mtpcache-eagleshift
[ -f "$ROLLBACK" ] || { log "ABORT: rollback launcher missing"; exit 3; }
grep -q "^DAILY_METHOD=mtp; DAILY_NS=3; DAILY_BLOCK=1472" /srv/qwen5090/launch-daily.sh || { log "ABORT: launch-daily.sh is not the MTP ns3 launcher"; exit 3; }
grep -q "^DAILY_IMG=$IMG " /srv/qwen5090/launch-daily.sh || { log "ABORT: launch-daily.sh DAILY_IMG is not the MTP image"; exit 3; }
grep -q 'SM12X PCIe IPC: MTP drafter admitted' /srv/qwen5090/launch-daily.sh || { log "ABORT: the 0148 assert is missing from the launcher"; exit 3; }
grep -q 'SM12X eagle-drop replay boundary retained' /srv/qwen5090/launch-daily.sh || { log "ABORT: the 0158 assert is missing from the launcher"; exit 3; }
grep -q "^DAILY_NS=7" /srv/qwen5090/launch-daily-r197-dflash-ns7-0906.sh || { log "ABORT: the frozen rollback is not the DFlash ns7 launcher"; exit 3; }
sudo docker image inspect "$IMG" >/dev/null 2>&1 || { log "ABORT: image $IMG missing"; exit 3; }
[ -f "$FD/corpus.jsonl" ] && [ -f "$DREF/dec-bf16-ctx30000.jsonl" ] && [ -f "$DREF/dec-bf16-ctx0.jsonl" ] || { log "ABORT: bf16 decode reference missing"; exit 3; }
for t in kv_capacity_probe.py needle_gate.sh decode_ss.py tooleval_summary.py decode_fidelity.py; do [ -f "$PR/$t" ] || { log "ABORT: $PR/$t missing"; exit 3; }; done
. /srv/qwen5090/lib/gpu-queue.sh
HAVE_LOCK=0; exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
rollback(){ teardown; log "ROLLBACK: booting the DFlash ns7 daily from $ROLLBACK (the tier is wiped again on the way back)"; env -i HOME="$HOME" USER="$USER" PATH="$PATH" bash $ROLLBACK > "$R/boot-rollback.log" 2>&1 && log "ROLLBACK daily up: $(grep -aoE 'Pool [0-9]+' "$R/boot-rollback.log" | tail -1)" || log "ROLLBACK FAILED too: $(grep -aE 'FAILED' "$R/boot-rollback.log" | tail -1 | cut -c1-200)"; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then curl -sf -m 5 $U/health >/dev/null || rollback; fi; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
HAVE_LOCK=1
log "=== R207 promote MTP ns3 start (lock held) ==="
teardown
log "native-l2 before: $(du -sh $L2 2>/dev/null | cut -f1), stamp '$(cat $L2/.block 2>/dev/null || echo none)' (a mismatch wipes _model_* inside the launcher)"
PIN=13980000000
if env -i HOME="$HOME" USER="$USER" PATH="$PATH" bash /srv/qwen5090/launch-daily.sh > "$R/boot-daily-$PIN.log" 2>&1 && curl -sf -m 5 $U/health >/dev/null; then :
else
  log "boot at the table pin FAILED: $(grep -aE 'FAILED' "$R/boot-daily-$PIN.log" | tail -1 | cut -c1-220)"
  teardown; PIN=13500000000; log "retrying with KV_BYTES=$PIN (R191: warmup 'invalid argument' flake at 13.98; launcher table must be updated if this pin sticks)"
  env -i HOME="$HOME" USER="$USER" PATH="$PATH" KV_BYTES=$PIN bash /srv/qwen5090/launch-daily.sh > "$R/boot-daily-$PIN.log" 2>&1 && curl -sf -m 5 $U/health >/dev/null || { log "boot at $PIN FAILED: $(grep -aE 'FAILED' "$R/boot-daily-$PIN.log" | tail -1 | cut -c1-220)"; rollback; exit 1; }
fi
log "DAILY UP pin=$PIN $(tail -1 "$R/boot-daily-$PIN.log" | cut -c1-300)"
grep -aE "block stamp|wiping" "$R/boot-daily-$PIN.log" | cut -c1-240 | sed 's/^/[wipe] /' | tee -a "$R/audit.log"
log "native-l2 after: $(du -sh $L2 2>/dev/null | cut -f1), stamp '$(cat $L2/.block 2>/dev/null || echo none)' (expected 1472)"
[ "$(cat $L2/.block 2>/dev/null)" = 1472 ] || log "WARN: block stamp is not 1472 after the boot"
BOOT=$(sudo docker logs vllm-27b 2>&1); echo "$BOOT" > "$R/engine-boot-daily.log"
log "[block] $(echo "$BOOT" | grep -aoE 'Setting attention block size to [0-9]+ tokens' | head -1)  [spec] $(echo "$BOOT" | grep -aoE 'num_spec_tokens=[0-9]+' | head -1)"
log "[mtp] $(echo "$BOOT" | grep -a 'MTP drafter admitted' | tail -1 | sed 's/^.*SM12X/SM12X/' | cut -c1-160)"
log "[0158] $(echo "$BOOT" | grep -a 'replay boundary retained' | tail -1 | sed 's/^.*SM12X/SM12X/' | cut -c1-200)"
log "[allreduce] $(echo "$BOOT" | grep -aoE "Using \[[^]]*\] all-reduce backends[^.]*" | head -1)"
log "[tier] $(echo "$BOOT" | grep -aoE 'primary tier \(lru, [0-9]+ blocks\)' | tail -1)  [artifact] $(echo "$BOOT" | grep -aoE 'torch_compile_cache/[0-9a-f]+/rank_0_0/backbone' | head -1)"
log "[layout] $(curl -s -m 5 $U/metrics | grep -aoE '^vllm:cache_config_info.*' | grep -oE 'block_size="[0-9]+"|num_gpu_blocks="[0-9]+"|kv_cache_size_tokens="[0-9]+"' | tr '\n' ' ') min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)MiB"
sleep 20
# The gate the route switch owes: the r173c bf16 decode ruler. ns9 sits at 0.0051–0.0062 median |Δlogprob| at 30K; ns7 (retracted) at 2×.
dfid(){ local ctx=$1
  python3 $PR/decode_fidelity.py run --url $U --corpus "$FD/corpus.jsonl" --out "$R/dec-daily-ctx$ctx.jsonl" --chunks 20 --ctx "$ctx" --tokens 256 > "$R/dec-daily-ctx$ctx.out" 2>&1
  log "[daily decode ctx$ctx vs bf16] $(python3 $PR/decode_fidelity.py compare "$DREF/dec-bf16-ctx$ctx.jsonl" "$R/dec-daily-ctx$ctx.jsonl" 2>&1 | tail -1 | cut -c1-300)"; }
dfid 0
dfid 30000
cap(){ log "[cap $1] $(python3 $PR/kv_capacity_probe.py --url $U "${@:2}" 2>&1 | tail -1 | cut -c1-330)"; }
cap short1 --ctx 0 --conc 1 --tokens 400 --ignore-eos --seed 31
cap ctx100k --ctx 120000 --conc 1 --tokens 200 --ignore-eos --seed 32
cap five100k --ctx 120000 --conc 5 --tokens 3000 --ignore-eos --seed 33
log "needle gate start (131K + 220K cold on the wiped tier, then the evicted re-asks through the tiers; EVICT=16 for the 1.31M pool)"
U=$U EVICT=16 bash $PR/needle_gate.sh post "$R" > "$R/needle-gate.log" 2>&1; rc=$?
log "needle gate rc=$rc: $(grep -aE 'SUMMARY|PASS|FAIL|tier_served' "$R/needle-gate.log" | tail -3 | tr '\n' ' ' | cut -c1-300)"
p1(){ local name=$1; shift
  python3 $PR/decode_ss.py --url $U --model qwen3.8-27b "$@" --out "$R/decode-$name.jsonl" > "$R/probe-$name.out" 2> "$R/probe-$name.err"
  if grep -aq RESULT "$R/probe-$name.out"; then grep -a RESULT "$R/probe-$name.out" | sed "s/^/[daily $name] /" | cut -c1-260 | tee -a "$R/audit.log"; else log "[daily $name] PROBE FAILED"; fi; }
p1 code-c1 --conc 1 --tokens 1024 --runs 3 --kind code
p1 prose-c1 --conc 1 --tokens 1024 --runs 2 --kind prose
p1 prose-c1-30k --conc 1 --tokens 1024 --runs 2 --kind prose --ctx 30000
p1 code-c8 --conc 8 --tokens 1024 --runs 2 --kind code
p1 prose-c8 --conc 8 --tokens 1024 --runs 2 --kind prose
p1 code-c16 --conc 16 --tokens 1024 --runs 2 --kind code
p1 prose-c16 --conc 16 --tokens 1024 --runs 2 --kind prose
( cd "$HOME" && tool-eval-bench --base-url $U/v1 --model qwen3.8-27b --temperature 0.6 --top-p 0.95 --top-k 20 --trials 4 --parallel 8 --json-file "$R/tooleval-daily.json" > "$R/tooleval-daily.log" 2>&1 )
python3 $PR/tooleval_summary.py "$R/tooleval-daily.json" daily-mtp-ns3 2>&1 | tee -a "$R/audit.log"
log "engine error lines: $(sudo docker logs vllm-27b 2>&1 | grep -ac 'illegal memory\|CUDA error\|Traceback\|OutOfMemoryError')  preemptions: $(curl -s -m 5 $U/metrics | grep -a '^vllm:num_preemptions_total' | awk '{print $NF}')"
grep -aE "DAILY UP|wipe\]|native-l2|block\]|mtp\]|0158\]|allreduce|tier\]|layout|decode ctx|cap |needle|RESULT|PROBE FAILED|tool-eval|error lines|FAILED|ROLLBACK|WARN" "$R/audit.log" | cut -c1-330 > "$R/sheet.txt"
log "=== R207 promote MTP DONE — daily = MTP ns3 (pin $PIN); rollback $ROLLBACK ==="
