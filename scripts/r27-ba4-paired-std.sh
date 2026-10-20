#!/usr/bin/env bash
# BA4 paired standard-benchmark check (2026-10-20). r27-ba4-std-bench (2026-10-19/20) read below r27-l1a-std-bench (10-17) from
# 2 streams up (ShareGPT -1.8..-6.8 %, Spec-Bench -1.4..-3.7 %) and TTFT p50 +7..+57 % in every cell, c1 included, while the
# paired decode screens at c4/c8 did not show it. Two rounds three days apart are not a pairing; this screen is.
# Arms on the SAME served image and launcher, EXP=1 :8029, BSS=1 as the daily: B1 = BA4 on (served), B0 = BA4=0 (the BA4 layer
# inactive: computes as L1a). Boot order B1 B0 B0 B1, eval-l2 wiped per boot. Per boot, ShareGPT V3 cells c1 (n 100),
# c8 (n 200), c16 (n 400) through the r27 std-bench client (vllm bench serve v0.30.0 + shim, T 0, same flags), one seed per
# cell shared by both arms so no cell reuses another cell's prefixes inside a boot. GPU ~45 min.
# Rule, written before the data: BA4 is cleared at a cell if median(B1)/median(B0) output tok/s >= 0.98 and TTFT p50 <= 1.05x;
#   a cell outside both bounds on both boot pairs is a confirmed BA4 regression at that concurrency.
# Round 2 (COLD=1, ARMS="1 0 0 1 1 0", CELLS="c1 c8"): a review of the boot logs found every
#   B1 boot loaded ONE warm AOT artifact and every B0 boot another (B0 compiled it cold at boot 2), so the paired result may be a
#   compile-lottery draw (R193) rather than BA4. Each boot here compiles cold into its own CACHE_DIR (torch_compile, inductor and
#   triton empty; flashinfer JIT copied), so each arm gets three independent artifacts. Rule, written before the data: if the c8
#   B1/B0 output ratio over the cold draws is >= 0.98, or the spread between draws of one arm exceeds the B1-B0 gap, the paired
#   regression is the artifact lottery, not BA4; if all three B1 draws are below all three B0 draws, BA4 itself costs c8.
# RUN (on the serving host, installed as /srv/qwen5090/ba4-paired-std/screen.sh):
#   sudo systemd-run --unit=r27-ba4-paired-std --collect -p RuntimeMaxSec=43200 -p TimeoutStopSec=900 -p Environment=HOME=$HOME bash /srv/qwen5090/ba4-paired-std/screen.sh
set -uo pipefail
export HOME=${HOME:?} USER=${USER:-$(id -un)}
export PATH=$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
D=/srv/qwen5090; H=$D/ba4-paired-std; U=http://127.0.0.1:8029
R=$D/results/$(date -u +%F-%H%M%S)-r27-ba4-paired-std; mkdir -p "$R/results" "$R/probes"
log(){ echo "$(date -Is) [r27-ba4-paired-std] $*" | tee -a "$R/audit.log"; }
echo "SUMMARY $(basename "$R") results $R" >> "$R/summary.txt"
[ "$(md5sum < $D/launch-daily.sh | cut -d' ' -f1)" = 7cb1a981b0d7db5a54dd8e03c52e1b09 ] || { log "ABORT: launch-daily.sh is not the BA4 daily"; exit 3; }
MDIR=$D/models/qwen3.8-27b-nvidia-nvfp4; DS=$D/datasets/std-bench; SG=$DS/ShareGPT_V3_unfiltered_cleaned_split.json
CLIENT_IMG=vllm/vllm-openai:v0.30.0
cp $D/probes/vllm_bench_tabby.py $D/probes/parse_container.py "$R/probes/"
CENV=(-e VLLM_NO_USAGE_STATS=1 -e VLLM_DO_NOT_TRACK=1 -e DO_NOT_TRACK=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
      -e HF_DATASETS_OFFLINE=1 -e HF_HUB_DISABLE_TELEMETRY=1 -e VLLM_USE_RUST_BENCH=0 -e TABBY_FORCE_LEN=1)
export GPU_QUEUE_NAME=r27-ba4-paired-std
. $D/lib/gpu-queue.sh
exec 9>$D/gpu-exclusive.lock; flock 9
teardown(){ sudo docker ps -a --format '{{.Names}}' | grep -qx vllm-exp || return 0
  sudo docker logs vllm-exp > "$R/engine-$1.log" 2>&1; sudo docker rm -f vllm-exp >/dev/null 2>&1
  sudo $D/shm-sweep.sh quiet >> "$R/shm-sweep.log" 2>&1 || true; sleep 5; }
finish(){ rc=$?; teardown end; rm -f "$GPU_QUEUE_MARK"; flock -u 9
  if [ -z "$(gpu_queue_others)" ]; then timeout -k 20 1380 bash $D/r858/restart-glm-daily.sh > "$R/glm-restore.log" 2>&1 \
    && log "GLM restored" || log "GLM RESTORE FAILED"; fi
  echo "SUMMARY $(basename "$R") $([ $rc = 0 ] && echo DONE || echo "FAILED rc=$rc")" | tee -a "$R/summary.txt"; }
trap finish EXIT
sudo docker stop -t 80 glm53 > "$R/glm-stop.log" 2>&1 || true
cell(){ local tag=$1 conc=$2 n=$3 seed=$4
  timeout -k 30 1800 sudo docker run --rm --name r27-paired-client --network host "${CENV[@]}" \
    -e TABBY_SAMPLES_OUT="/out/$tag.samples.tsv" \
    -v "$MDIR":/model:ro -v "$DS":/data:ro -v "$R/probes":/probes:ro -v "$R/results":/out \
    --entrypoint python3 "$CLIENT_IMG" /probes/vllm_bench_tabby.py bench serve \
    --backend openai-chat --base-url $U --endpoint /v1/chat/completions \
    --model qwen3.8-27b --tokenizer /model --temperature 0 \
    --request-rate inf --max-concurrency "$conc" --num-warmups 0 --num-prompts "$n" --no-oversample \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-dir /out --result-filename "$tag.json" --disable-tqdm \
    --dataset-name sharegpt --dataset-path "/data/$(basename "$SG")" --seed "$seed" > "$R/client-$tag.log" 2>&1
  log "[$tag] rc=$? $(python3 -c 'import json,sys;j=json.load(open(sys.argv[1]));print("out %.1f tok/s ttft p50 %.1f ms tpot p50 %.2f ms ok %d"%(j["output_throughput"],j["median_ttft_ms"],j["median_tpot_ms"],j["completed"]))' "$R/results/$tag.json" 2>&1)"; }
ARMS=${ARMS:-"1 0 0 1"}; CELLS=${CELLS:-"c1 c8 c16"}; COLD=${COLD:-0}
log "arms $ARMS cells $CELLS cold $COLD"
i=0
for arm in $ARMS; do
  i=$((i+1)); b="$i-B$arm"
  sudo find $D/eval-l2 -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} +
  log "boot $b (BA4=$arm)"
  CX=""
  if [ "$COLD" = 1 ]; then CX=$R/cache-$b; mkdir -p "$CX/torch_compile_qwen38_v0280nv" "$CX/inductor" "$CX/triton"; cp -a $D/cache/flashinfer "$CX/"
    CX="DAILY_ALLOW_ENV=1 CACHE_DIR=$CX"; fi
  if ! env -i PATH="$PATH" HOME="$HOME" USER="$USER" EXP=1 SEQS=16 BSS=1 BA4=$arm $CX bash $D/launch-daily.sh > "$R/boot-$b.log" 2>&1; then
    log "boot $b FAILED: $(tail -1 "$R/boot-$b.log")"; teardown "$b"; exit 1; fi
  log "UP $b: $(tail -1 "$R/boot-$b.log" | grep -ao 'Pool [0-9]*\|BA4 [0-9]*/96 active' | tr '\n' ' ')"
  curl -sf -m 120 $U/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Say hi."}],"max_tokens":16}' >/dev/null || log "[$b] warm-up failed"
  case " $CELLS " in *" c1 "*) cell "$b-c1" 1 100 7311;; esac
  case " $CELLS " in *" c8 "*) cell "$b-c8" 8 200 7318;; esac
  case " $CELLS " in *" c16 "*) cell "$b-c16" 16 400 7326;; esac
  log "[$b] compile: $(sudo docker logs vllm-exp 2>&1 | grep -aoE 'Compiling a graph for compile range|Directly load|num_artifacts=[0-9]+|torch.compile took [0-9.]+ s' | sort | uniq -c | tr '\n' ' ')"
  teardown "$b"; [ "$COLD" = 1 ] && rm -rf "$R/cache-$b"
done
python3 - "$R" "$ARMS" "$CELLS" <<'PY' | tee "$R/verdict.txt"
import json, statistics as st, sys
from pathlib import Path
r = Path(sys.argv[1]); j = lambda t: json.load(open(r / "results" / f"{t}.json"))
arms = sys.argv[2].split(); cells = [int(c[1:]) for c in sys.argv[3].split()]
for c in cells:
    a = {arm: [j(f"{i}-B{arm}-c{c}") for i, x in enumerate(arms, 1) if x == arm] for arm in "10"}
    print(f"c{c} draws: B1 " + " ".join(f"{x['output_throughput']:.1f}" for x in a["1"]) + " | B0 " + " ".join(f"{x['output_throughput']:.1f}" for x in a["0"]))
    o = {k: st.median(x["output_throughput"] for x in v) for k, v in a.items()}
    t = {k: st.median(x["median_ttft_ms"] for x in v) for k, v in a.items()}
    pairs = [(a["1"][k]["output_throughput"] / a["0"][k]["output_throughput"], a["1"][k]["median_ttft_ms"] / a["0"][k]["median_ttft_ms"]) for k in range(min(len(a["1"]), len(a["0"])))]
    ok = o["1"] / o["0"] >= 0.98 and t["1"] / t["0"] <= 1.05
    bad = all(p < 0.98 and q > 1.05 for p, q in pairs)
    print(f"c{c}: out B1 {o['1']:.1f} B0 {o['0']:.1f} ({100*(o['1']/o['0']-1):+.1f} %), ttft p50 B1 {t['1']:.1f} B0 {t['0']:.1f} ms "
          f"({100*(t['1']/t['0']-1):+.1f} %) -> {'CLEARED' if ok else ('BA4 REGRESSION' if bad else 'UNCLEAR')}")
PY
echo "SUMMARY $(basename "$R") $(tr '\n' ' ' < "$R/verdict.txt")" >> "$R/summary.txt"
