#!/usr/bin/env bash
# r27-headroom-stress (2026-10-20): does a raised KV pin leave enough VRAM for what the pin ladder did not exercise?
# r27-pin-ladder qualified 16.90 GB/GPU (pool 1,583,674, +13.8 %) on a text-only stress (five 120K + one 250K), worst
# boot 1,181 MiB free. With the pool pinned (--kv-cache-memory-bytes) vLLM does not size anything from its profile run,
# so serve-time peaks the ladder never produced -- vision-encoder activations (the checkpoint's processor allows
# 16,384-token images; R161) and all 16 slots decoding during a long prefill -- come out of whatever the pin leaves.
# One boot per pin, tier wiped, EXP :8029 with the daily's PCIE_IPC=1 BSS=1 SEQS=16, stock power. Concurrent load:
#   16 code streams decoding 1,024 tokens (decode_ss), a 120K-token cold prefill (kv_capacity_probe), the 16 R161
#   screenshots in one request, and 4 synthetic 4096x4096 noise images (max-size, ~16K tokens each) in one request.
# Free VRAM (min over both cards) sampled every 0.5 s.
# Rule, written before the data: a pin passes iff every request succeeds, the engine logs no OOM / CUDA error, and the
# minimum free VRAM stays >= 1,000 MiB. 16.90 runs first; 15.90 runs only if 16.90 fails. Daily restored at the end.
#   ssh flan 'sudo systemd-run --unit=r27-headroom-stress --collect -p User="$(id -un)" -p RuntimeMaxSec=7200 -p TimeoutStopSec=900 -E HOME=$HOME -E GPU_QUEUE_NAME=r27-headroom-stress bash /srv/qwen5090/r27-headroom-stress.sh'
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/$(date -u +%F-%H%M%S)-r27-headroom-stress; mkdir -p "$R"
log(){ echo "$(date -Is) [r27-headroom-stress] $*" | tee -a "$R/audit.log"; }
U=http://127.0.0.1:8029; PR=/srv/qwen5090/probes; L2=/srv/qwen5090/eval-l2
IMG12=/srv/qwen5090/results/2026-09-03-r161-images/img
case "$(md5sum < /srv/qwen5090/launch-daily.sh | cut -c1-8)" in e680228b|"${WANT_MD5:-none}") ;; *) log "ABORT: launch-daily.sh is not the expected daily"; exit 3;; esac
. /srv/qwen5090/lib/gpu-queue.sh
exec 9>/srv/qwen5090/gpu-exclusive.lock; flock 9
settle(){ for i in $(seq 36); do busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{c++} END{print c+0}'); [ "$busy" = 0 ] && break; sleep 5; done; sleep 45; }
teardown(){ for c in vllm-27b vllm-exp; do sudo docker ps -a --format '{{.Names}}' | grep -qx "$c" || continue
  [ "$c" = vllm-exp ] && sudo docker logs vllm-exp > "$R/engine-${CUR:-x}.log" 2>&1; sudo docker rm -f "$c" >/dev/null 2>&1; done; settle; }
finish(){ rm -f "${GPU_QUEUE_MARK:-/nonexistent}"; teardown; flock -u 9
  bash /srv/qwen5090/daily-restore-retry.sh 2>&1 | grep -aE "DAILY|FAILED|attempt|SKIPPED" | cut -c1-160 | tee -a "$R/audit.log"
  log "=== $1 ==="; echo "SUMMARY $(basename "$R") $1" >> "$R/summary.txt"; }
trap 'finish ABORTED; exit 4' TERM
bash /srv/qwen5090/daily-power.sh stock 2>&1 | grep -aiE '^WARN' | tee -a "$R/audit.log"
# four max-size noise PNGs, stdlib only (no PIL on the host)
python3 - "$R/big" <<'PY'
import os, sys, zlib, struct
d = sys.argv[1]; os.makedirs(d, exist_ok=True)
def png(path, w, h, seed):
    rnd = __import__("random").Random(seed)
    raw = b"".join(b"\x00" + rnd.randbytes(w * 3) for _ in range(h))
    def chunk(t, b): return struct.pack(">I", len(b)) + t + b + struct.pack(">I", zlib.crc32(t + b) & 0xffffffff)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(raw, 1)) + chunk(b"IEND", b""))
for i in range(4): png(f"{d}/big{i}.png", 4096, 4096, i)
PY
ls -la "$R/big" | tail -4 | tee -a "$R/audit.log"
mmreq(){ python3 - "$1" "$2" "$3" <<'PY'
import base64, glob, json, sys, time, urllib.request
url, d, tag = sys.argv[1], sys.argv[2], sys.argv[3]
files = sorted(glob.glob(d + "/*.png"))
content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(open(f, "rb").read()).decode()}} for f in files]
content.append({"type": "text", "text": "Describe each image in one short sentence."})
body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": content}], "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False}}
t0 = time.time()
try:
    r = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                                                {"Content-Type": "application/json"}), timeout=900))
    print("RESULT " + json.dumps({"tag": tag, "ok": True, "images": len(files), "s": round(time.time() - t0, 1), "usage": r["usage"]}))
except Exception as e:
    body = getattr(e, "read", lambda: b"")()[:400]
    print("RESULT " + json.dumps({"tag": tag, "ok": False, "images": len(files), "err": repr(e)[:200], "body": body.decode(errors="replace")}))
PY
}
run(){ local kvb=$1; CUR=$kvb
  log "--- pin $kvb: tier wiped, boot ---"
  sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + 2>/dev/null; sync
  if ! env -i PATH="$PATH" HOME="$HOME" USER="$(id -un)" EXP=1 SEQS=16 BSS=1 PCIE_IPC=1 KV_BYTES="$kvb" POOL_MIN=100000 POOL_MAX=1900000 \
       bash /srv/qwen5090/launch-daily.sh > "$R/boot-$kvb.log" 2>&1; then
    log "[$kvb] BOOT FAILED: $(grep -aoE 'CUDA error: [a-z ]+|FAILED: [^,]{0,90}' "$R/boot-$kvb.log" | tail -1)"; teardown; return 1; fi
  log "[$kvb] UP: $(tail -1 "$R/boot-$kvb.log" | grep -aoE 'Pool [0-9]+, min free VRAM [0-9]+ MiB|BA4 [0-9]+/96 active' | tr '\n' ' ')"
  : > "$R/vram-$kvb.txt"; ( while :; do nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1 >> "$R/vram-$kvb.txt"; sleep 0.5; done ) & local SP=$!
  local N=$(( $(date +%s) % 100000 ))
  python3 $PR/decode_ss.py --url $U --model qwen3.8-27b --kind code --conc 16 --tokens 1024 --runs 1 --seed-prefix "hs$N-" \
    --out "$R/c16-$kvb.jsonl" > "$R/c16-$kvb.out" 2>&1 & local P1=$!
  sleep 8
  python3 $PR/kv_capacity_probe.py --url $U --ctx 120000 --conc 1 --tokens 64 --seed $(( (N + 7) % 100000 )) > "$R/prefill-$kvb.out" 2>&1 & local P2=$!
  mmreq $U "$IMG12" "img12" > "$R/img12-$kvb.out" 2>&1 & local P3=$!
  mmreq $U "$R/big" "big4" > "$R/big4-$kvb.out" 2>&1 & local P4=$!
  wait $P1 $P2 $P3 $P4
  kill $SP 2>/dev/null; wait $SP 2>/dev/null
  local minfree; minfree=$(sort -n "$R/vram-$kvb.txt" | head -1)
  sudo docker logs vllm-exp > "$R/engine-$kvb.log" 2>&1
  local oom; oom=$(grep -acE "OutOfMemory|CUDA out of memory|CUDA error" "$R/engine-$kvb.log")
  local ok=1
  grep -aq RESULT "$R/c16-$kvb.out" || ok=0
  grep -aq '"preemptions"' "$R/prefill-$kvb.out" || ok=0
  grep -aq '"ok": true' "$R/img12-$kvb.out" || ok=0
  grep -aq '"ok": true' "$R/big4-$kvb.out" || ok=0
  log "[$kvb] c16: $(grep -a RESULT "$R/c16-$kvb.out" | grep -aoE '"ss_agg_tps_median": [0-9.]+' | head -1)  prefill: $(tail -1 "$R/prefill-$kvb.out" | cut -c1-120)"
  log "[$kvb] img12: $(grep -a RESULT "$R/img12-$kvb.out" | cut -c1-200)"
  log "[$kvb] big4: $(grep -a RESULT "$R/big4-$kvb.out" | cut -c1-260)"
  log "[$kvb] min free VRAM ${minfree:-?} MiB over $(wc -l < "$R/vram-$kvb.txt") samples, engine OOM/CUDA-error lines: $oom, all requests ok: $ok"
  CUR=$kvb; teardown
  [ "$ok" = 1 ] && [ "$oom" = 0 ] && [ "${minfree:-0}" -ge 1000 ]; }
teardown
# 2026-10-20 second use (user: "would be nice if we hit/went right above 1.5M kv"): PINS="16020000000 16020000000" runs the
# same stress on two boots of one pin; the verdict then needs every boot to pass. Default keeps the first run's sequence.
if [ -n "${PINS:-}" ]; then V=PASS; for p in $PINS; do run "$p" || V=FAIL; done; V="$V pins [$PINS]"
elif run 16900000000; then V="PASS 16.90 GB"; elif run 15900000000; then V="PASS 15.90 GB (16.90 failed)"; else V="FAIL both (keep 14.86)"; fi
log "VERDICT $V"
finish "$V"
