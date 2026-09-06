#!/usr/bin/env bash
# R204 (2026-09-06, R202 §6 steps 3-4, user "ok go"): gate the two cherry-pick images built by build-r204-picks.sh on the daily image
# (vllm-qwen38:…-pcieipc-bsshash), EXP=1 on :8029 (eval-l2 tier), same route as the daily (SEQS 16, PCIE_IPC=1, BSS=1, dflash ns7).
#   PICKS arm (0149 #55507 align seed, 0150 #54275 kv-suggestion clamp, 0151 #54972 UVA empty_cache, 0152 #52807 offload load boundary):
#     boots 1-4 = boot-only at the 13.98 GB pin (Bug C rate on this image; R191 saw 2 of 3 boots die in the Triton warmup at that pin),
#     each logging pool / min_free / the "fully utilize" suggestion (the only thing 0150 changes when the pin is given) / aot artifact;
#     boot 5 stays up for: cache layout, kv_capacity short/100K/five×100K, warm revisit (32K same prompt twice: GPU prefix hit) and a
#     tier revisit (32K, 12×90K flood between the sends: the offloading connector must serve it — 0152's territory), needle gate
#     (131K + 220K cold + evicted re-asks through the tiers), decode_ss code-c8 / prose-c1 / code-c16, dense + agentic rulers vs bf16
#     AND per-doc vs the R203 ON dump (same code path expected: the two-boot floor applies), error lines.
#   GDNFI arm (0153 #50862, FlashInfer SM120 CuTe-DSL GDN prefill instead of Triton/FLA — a NEW PREFILL KERNEL CLASS, built alone on the
#     daily image): one boot; assert "Using FlashInfer GDN prefill kernel"; prefill tok/s at 2K/4K/8K/32K/131K (needle_depth cold rows,
#     the R167 instrument) vs the same rows on the PICKS arm (Triton/FLA); needle hits; dense + agentic rulers vs bf16 (the ladder IS
#     a prefill measurement, so this is the kernel's fidelity read) + per-doc vs R203 ON; decode_ss code-c8 / prose-c1 (decode path
#     unchanged, sanity); error lines. The compile artifact almost certainly differs (new custom op in the graph), so no bitwise pair:
#     fidelity is judged against bf16, with the PICKS/R203-ON numbers as the same-day daily reference.
# Verdicts are written to the sheet, not decided here: promotion is the user's call.
#   unit: sudo systemd-run --unit=r204-picks-gate --collect -p User=adrienbrault -p RuntimeMaxSec=43200 -p TimeoutStopSec=900 \
#         -E GPU_QUEUE_NAME=r204-picks-gate bash /srv/qwen5090/r204-picks-gate.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-09-06-r204-picks-gate; mkdir -p "$R"
log(){ echo "$(date -Is) $*" | tee -a "$R/audit.log"; }
U=http://127.0.0.1:8029; CAND=/srv/qwen5090/launch-daily.sh; L2=/srv/qwen5090/eval-l2; PR=/srv/qwen5090/probes
BF16_DIR=/srv/qwen5090/results/2026-09-01-r156-bf16-ladder; BF16_REF=$BF16_DIR/dump-a1-dense.jsonl; LADDER_CORPUS=/srv/qwen5090/r156-corpus.jsonl
R203=/srv/qwen5090/results/2026-09-06-r203-spec-ladder
DAILY=$(sed -nE 's/^DAILY_IMG=([^ ]+).*/\1/p' "$CAND" | head -1); PICKS=$DAILY-picks; GDNFI=$DAILY-gdnfi
PIN=13980000000; PIN_FALLBACK=13500000000
[ -n "$DAILY" ] || { log "ABORT: cannot read DAILY_IMG"; exit 3; }
for i in "$PICKS" "$GDNFI"; do sudo docker image inspect "$i" >/dev/null 2>&1 || { log "ABORT: image $i missing (build-r204-picks.sh)"; exit 3; }; done
sudo docker run --rm --entrypoint cat "$PICKS" /opt/prs-markers/0152 2>/dev/null | grep -q PRS-0152 || { log "ABORT: $PICKS lacks the 0152 marker"; exit 3; }
sudo docker run --rm --entrypoint cat "$GDNFI" /opt/prs-markers/0153 2>/dev/null | grep -q PRS-0153 || { log "ABORT: $GDNFI lacks the 0153 marker"; exit 3; }
[ -f "$BF16_REF" ] && [ -f "$LADDER_CORPUS" ] && [ -f "$BF16_DIR/dump-a1-agentic.jsonl" ] && [ -f "$BF16_DIR/agentic-ids.jsonl" ] || { log "ABORT: reference dumps/corpus missing"; exit 3; }
grep -q "R183 EXP-only passthrough" "$CAND" || { log "ABORT: launch-daily.sh lacks the R183 EXP passthrough"; exit 3; }
for t in fidelity_ladder.py fidelity_compare.py agentic_ref.py ladder_doc_compare.py decode_ss.py kv_capacity_probe.py needle_gate.sh needle_depth.py warm-revisit.py; do [ -f "$PR/$t" ] || { log "ABORT: $PR/$t missing"; exit 3; }; done
. /srv/qwen5090/lib/gpu-queue.sh
HAVE_LOCK=0; exec 9>/srv/qwen5090/gpu-exclusive.lock
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep "${1:-60}"; }
teardown(){ for c in vllm-27b vllm-exp vllm-eval; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue; sudo docker logs "$c" > "$R/engine-$c-$(date +%H%M%S).log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ teardown; log "restoring daily (skipped if another unit is queued: $(gpu_queue_others | tr '\n' ' '))"; bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|KV pool|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"; log "=== R204 $1 ==="; }
trap 'log "### SIGTERM ###"; if [ "$HAVE_LOCK" = 1 ]; then finish ABORTED; else log "no lock held: engines left alone, exiting"; fi; exit 4' TERM
flock -n 9 || { log "waiting for the GPU-exclusive lock (another unit holds it)"; flock 9; }
HAVE_LOCK=1
log "=== R204 start (lock held): PICKS=$PICKS GDNFI=$GDNFI on the daily route (EXP=1 SEQS=16 PCIE_IPC=1 BSS=1 dflash ns7) ==="
mountpoint -q "$L2" || sudo bash /srv/qwen5090/eval-l2-dio.sh || { log "FAILED: eval-l2 not mounted"; finish ABORTED; exit 1; }
wipe_l2(){ sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync; }
ELOG(){ sudo docker logs vllm-exp 2>&1; }
errs(){ ELOG | grep -ac 'illegal memory\|CUDA error\|Traceback\|OutOfMemoryError\|JointFailure'; }
# boot_once TAG IMG PIN → 0 = up
boot_once(){ local tag=$1 img=$2 kv=$3 rc; 
  env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=16 KV_BYTES=$kv PCIE_IPC=1 BSS=1 CAND_IMG=$img EXTRA_ENV_APPEND="-e VLLM_TRITON_FORCE_FIRST_CONFIG=1" bash $CAND > "$R/boot-$tag-$kv.log" 2>&1; rc=$?
  if [ $rc -eq 0 ] && curl -sf -m 5 $U/health >/dev/null; then
    ELOG > "$R/engine-boot-$tag.log"
    local saved loaded; saved=$(grep -ac 'saved AOT compiled function' "$R/engine-boot-$tag.log"); loaded=$(grep -ac 'Directly load AOT' "$R/engine-boot-$tag.log")
    log "[$tag] BOOT OK pin=$kv pool=$(grep -aoE 'Pool [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9) min_free=$(grep -aoE 'min free VRAM [0-9]+' "$R/boot-$tag-$kv.log" | tail -1 | tr -dc 0-9)MiB block=$(grep -aoE 'Setting attention block size to [0-9]+' "$R/engine-boot-$tag.log" | head -1 | tr -dc 0-9) suggest=$(grep -aoE 'fully utilize[^0-9]*[0-9.]+ ?GiB' "$R/engine-boot-$tag.log" | head -1 | grep -oE '[0-9.]+ ?GiB' | head -1) clamp_lines=$(grep -ac 'lowering the fully-utilize suggestion' "$R/engine-boot-$tag.log") gdn_prefill=$(grep -aoE 'Using [A-Za-z/]+ GDN prefill kernel' "$R/engine-boot-$tag.log" | head -1 | awk '{print $2}') aot_saved=$saved aot_loaded=$loaded $([ "$saved" = 0 ] && [ "$loaded" -ge 1 ] && echo SAME-ARTIFACT || echo FRESH-COMPILE) compile_hashes=$(grep -aoE 'torch_aot_compile/[0-9a-f]{12}' "$R/engine-boot-$tag.log" | cut -d/ -f2 | sort -u | tr '\n' ',')"
    return 0; fi
  log "[$tag] boot pin=$kv FAILED rc=$rc: $(grep -aE 'FAILED' "$R/boot-$tag-$kv.log" | tail -1 | cut -c1-220)"
  ELOG 2>/dev/null | grep -aiE "error|exception|Bug C|headroom" | head -5 | cut -c1-220 | sed "s/^/[$tag boot-err] /" | tee -a "$R/audit.log"
  ELOG > "$R/engine-bootfail-$tag.log" 2>/dev/null; teardown; return 1; }
layout(){ log "[$1 layout] $(curl -s -m 5 $U/metrics | grep -aoE '^vllm:cache_config_info.*' | grep -oE 'block_size="[0-9]+"|num_gpu_blocks="[0-9]+"|kv_cache_size_tokens="[0-9]+"' | tr '\n' ' ') min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)MiB"; }
cap(){ log "[$1 cap $2] $(python3 $PR/kv_capacity_probe.py --url $U "${@:3}" 2>&1 | tail -1 | cut -c1-330)"; }
p1(){ local tag=$1 name=$2; shift 2
  python3 $PR/decode_ss.py --url $U --model qwen3.8-27b "$@" --out "$R/decode-$tag-$name.jsonl" > "$R/probe-$tag-$name.out" 2> "$R/probe-$tag-$name.err"
  if grep -aq RESULT "$R/probe-$tag-$name.out"; then grep -a RESULT "$R/probe-$tag-$name.out" | sed "s/^/[$tag $name] /" | cut -c1-260 | tee -a "$R/audit.log"
  else log "[$tag $name] PROBE FAILED: $(grep -a . "$R/probe-$tag-$name.out" "$R/probe-$tag-$name.err" 2>/dev/null | tail -1 | cut -c1-140)"; fi; }
revisit(){ local tag=$1 name=$2; shift 2
  python3 $PR/warm-revisit.py --url $U --model qwen3.8-27b --ctx 32000 "$@" > "$R/revisit-$tag-$name.log" 2>&1
  log "[$tag revisit $name] $(grep -a RESULT "$R/revisit-$tag-$name.log" | cut -c1-300)"; }
needles_tps(){ local tag=$1; shift 1   # cold rows → prefill tok/s per depth (R167 instrument)
  python3 $PR/needle_depth.py --url $U/v1 --model qwen3.8-27b --depths "$@" --samples 2 --out "$R/needles-tps-$tag.jsonl" > "$R/needles-tps-$tag.out" 2>&1
  log "[$tag needles $*] $(grep -a SUMMARY "$R/needles-tps-$tag.out" | cut -c1-120) prefill tok/s per row: $(python3 -c "import json;print(' '.join(f\"{r['depth']//1000}K:{r['prompt_tokens']/r['cold_s']:.0f}\" for r in map(json.loads,open('$R/needles-tps-$tag.jsonl')) if 'cold_s' in r and r.get('prompt_tokens')))" 2>&1 | cut -c1-200)"; }
ruler_dense(){ local T=$1
  timeout 3600 python3 $PR/fidelity_ladder.py --url $U --model qwen3.8-27b --corpus "$LADDER_CORPUS" --out "$R/dump-$T-dense.jsonl" --logprobs 20 --mode dense --resume > "$R/score-$T-dense.out" 2>&1
  curl -sf -m 5 $U/health >/dev/null || { log "ENGINE DIED during the dense ruler: $(sudo dmesg -T | grep -a Xid | tail -1 | cut -c1-160)"; return 1; }
  log "[$T dense] docs failed=$(grep -ac '^\[warn\]' "$R/score-$T-dense.out") records=$(wc -l < "$R/dump-$T-dense.jsonl")"
  python3 $PR/fidelity_compare.py --ref "$BF16_REF" --arm "$R/dump-$T-dense.jsonl" --label "$T" --json "$R/fidelity-$T-dense.json" 2>&1 | tee "$R/fidelity-$T-dense.txt" | grep -aE "overall top-1|corpus PPL|truncated KL" | cut -c1-200 | sed "s/^/[$T dense vs bf16] /" | tee -a "$R/audit.log"
  [ -s "$R203/dump-ON-dense.jsonl" ] && python3 $PR/ladder_doc_compare.py --a "$R/dump-$T-dense.jsonl" --b "$R203/dump-ON-dense.jsonl" --label-a $T --label-b R203ON 2>&1 | tee "$R/doc-compare-dense-$T-vs-R203ON.txt" | grep -aE "^corpus|^per-doc|^positions|^DOC-COMPARE" | cut -c1-260 | sed "s/^/[$T dense per-doc vs R203 ON] /" | tee -a "$R/audit.log"; }
ruler_agentic(){ local T=$1
  timeout 3600 python3 $PR/agentic_ref.py score --url $U --model qwen3.8-27b --ids "$BF16_DIR/agentic-ids.jsonl" --out "$R/dump-$T-agentic.jsonl" > "$R/score-$T-agentic.out" 2>&1
  curl -sf -m 5 $U/health >/dev/null || { log "ENGINE DIED during the agentic ruler"; return 1; }
  python3 $PR/fidelity_compare.py --ref "$BF16_DIR/dump-a1-agentic.jsonl" --arm "$R/dump-$T-agentic.jsonl" --label "AGENTIC-$T" --json "$R/bf16-$T-agentic.json" 2>&1 | tee "$R/bf16-$T-agentic.txt" | grep -aE "overall top-1|corpus PPL" | cut -c1-200 | sed "s/^/[$T agentic vs bf16] /" | tee -a "$R/audit.log"
  [ -s "$R203/dump-ON-agentic.jsonl" ] && python3 $PR/ladder_doc_compare.py --a "$R/dump-$T-agentic.jsonl" --b "$R203/dump-ON-agentic.jsonl" --label-a $T --label-b R203ON 2>&1 | tee "$R/doc-compare-agentic-$T-vs-R203ON.txt" | grep -aE "^corpus|^per-doc|^positions|^DOC-COMPARE" | cut -c1-260 | sed "s/^/[$T agentic per-doc vs R203 ON] /" | tee -a "$R/audit.log"; }

### PICKS arm
teardown; wipe_l2
ok=0; fail=0
for i in 1 2 3 4; do
  if boot_once PICKS$i "$PICKS" $PIN; then ok=$((ok+1)); layout PICKS$i; teardown; else fail=$((fail+1)); fi
done
log "[PICKS] boots 1-4 at the 13.98 pin: ok=$ok failed=$fail"
if boot_once PICKS5 "$PICKS" $PIN; then ok=$((ok+1)); PICKS_PIN=$PIN
elif boot_once PICKS5b "$PICKS" $PIN_FALLBACK; then PICKS_PIN=$PIN_FALLBACK; log "[PICKS] boot 5 needed the 13.5 fallback"
else log "[PICKS] ARM FAILED: boot 5 failed at both pins"; PICKS_PIN=""; fi
log "[PICKS] 13.98-pin boot tally: ok=$ok of 5 (R191 daily-image reference: 1 of 3)"
if [ -n "$PICKS_PIN" ]; then
  sleep 20; layout PICKS
  cap PICKS short1 --ctx 0 --conc 1 --tokens 400 --ignore-eos --seed 31
  cap PICKS ctx100k --ctx 120000 --conc 1 --tokens 200 --ignore-eos --seed 32
  cap PICKS five100k --ctx 120000 --conc 5 --tokens 3000 --ignore-eos --seed 33
  revisit PICKS gpu-warm
  revisit PICKS tier-flood --flood 12 --flood-ctx 90000
  U=$U bash $PR/needle_gate.sh picks "$R" > "$R/needle-gate-PICKS.log" 2>&1; rc=$?
  log "[PICKS needle gate] rc=$rc: $(grep -aE 'SUMMARY' "$R/needle-gate-PICKS.log" | tail -1 | cut -c1-300)"
  needles_tps PICKS 2000 4000 8000 32000
  p1 PICKS code-c8 --conc 8 --tokens 1024 --runs 2 --kind code
  p1 PICKS prose-c1 --conc 1 --tokens 1024 --runs 2 --kind prose
  p1 PICKS code-c16 --conc 16 --tokens 1024 --runs 1 --kind code
  ruler_dense PICKS && ruler_agentic PICKS
  log "[PICKS] engine error lines: $(errs)  preemptions: $(curl -s -m 5 $U/metrics | grep -a '^vllm:num_preemptions_total' | awk '{print $NF}')  tier: $(curl -s -m 5 $U/metrics | grep -aE '^vllm:kv_offload_tiering_(fs|chunk|block)' | awk '{s=s" "$1"="$2} END{print s}' | cut -c1-200)"
  ELOG > "$R/engine-PICKS-final.log"
fi

### GDNFI arm
teardown; wipe_l2
if boot_once GDNFI "$GDNFI" $PIN || boot_once GDNFIb "$GDNFI" $PIN_FALLBACK; then
  sleep 20; layout GDNFI
  if grep -aq "Using FlashInfer GDN prefill kernel" "$R/engine-boot-GDNFI.log" "$R/engine-boot-GDNFIb.log" 2>/dev/null; then log "[GDNFI] FlashInfer GDN prefill kernel ACTIVE"; else log "[GDNFI] WARNING: FlashInfer GDN prefill kernel NOT selected (0153 inert?) — arm measures nothing new"; fi
  cap GDNFI short1 --ctx 0 --conc 1 --tokens 400 --ignore-eos --seed 31
  cap GDNFI ctx100k --ctx 120000 --conc 1 --tokens 200 --ignore-eos --seed 32
  needles_tps GDNFI 2000 4000 8000 32000 131000
  U=$U bash $PR/needle_gate.sh gdnfi "$R" > "$R/needle-gate-GDNFI.log" 2>&1; rc=$?
  log "[GDNFI needle gate] rc=$rc: $(grep -aE 'SUMMARY' "$R/needle-gate-GDNFI.log" | tail -1 | cut -c1-300)"
  ruler_dense GDNFI && ruler_agentic GDNFI
  p1 GDNFI code-c8 --conc 8 --tokens 1024 --runs 2 --kind code
  p1 GDNFI prose-c1 --conc 1 --tokens 1024 --runs 2 --kind prose
  log "[GDNFI] engine error lines: $(errs)  preemptions: $(curl -s -m 5 $U/metrics | grep -a '^vllm:num_preemptions_total' | awk '{print $NF}')"
  ELOG > "$R/engine-GDNFI-final.log"
else log "[GDNFI] ARM FAILED: could not boot at either pin"; fi
grep -aE "BOOT OK|boot pin|boots 1-4|tally|ARM FAILED|layout|cap |revisit|needle|prefill tok/s|RESULT|PROBE FAILED|vs bf16|per-doc|ACTIVE|NOT selected|error lines|ENGINE DIED|FAILED" "$R/audit.log" | cut -c1-330 > "$R/sheet.txt"
finish DONE
