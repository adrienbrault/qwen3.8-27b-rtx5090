#!/usr/bin/env bash
# NVFP4-KV gauntlet driver (patches-nvfp4kv/README.md steps 1-6) on :8029 / vllm-exp.
# Three engine generations, in order: NS=4 (full battery) -> NS=0 (needles + sean gate only)
# -> LINEAR_VSF=0 (needles MUST FAIL; pass = layout theory wrong, stop).
# Full raw capture under $R. Refuses while the DeepSWE verify unit or the daily is up.
#   sudo systemd-run --unit=nvfp4kv-gauntlet --collect -p User=adrienbrault bash /srv/qwen5090/nvfp4kv-gauntlet.sh
#   tail -f /srv/qwen5090/results/2026-08-21-qwen38-nvfp4kv/gauntlet.log ; stop: sudo systemctl stop nvfp4kv-gauntlet
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/srv/qwen5090/results/2026-08-21-qwen38-nvfp4kv; mkdir -p "$R"
URL=http://localhost:8029/v1; M=qwen3.8-27b
P=/srv/qwen5090/probes/needle_depth.py
log() { echo "$(date -Is) $*" | tee -a "$R/gauntlet.log"; }
systemctl is-active --quiet deepswe-verify && { log "REFUSE: deepswe-verify unit active"; exit 1; }
sudo docker ps --format '{{.Names}}' | grep -qE '^(vllm-27b|vllm-lmcache)$' && { log "REFUSE: daily container up"; exit 1; }

boot() { # $1=NS $2=LINEAR_VSF $3=tag
  log "### BOOT NS=$1 LINEAR_VSF=$2 ($3) ###"
  sudo docker rm -f vllm-eval vllm-exp >/dev/null 2>&1 || true
  NS=$1 LINEAR_VSF=$2 bash /srv/qwen5090/launch-qwen38-nvfp4kv.sh > "$R/boot-$3.log" 2>&1
  rc=$?; sudo docker logs vllm-exp > "$R/vllm-$3.log" 2>&1
  grep -aE "GPU KV cache size|Maximum concurrency|kv_cache_dtype|overlay|Using .* backend|block_size|num_speculative|swizzled" "$R/vllm-$3.log" | head -20 | tee -a "$R/gauntlet.log"
  return $rc
}
needles() { # $1=tag ; cold+warm, MTP-sensitive depths
  log "--- needles ($1) ---"
  python3 $P --url $URL --model $M --depths 9000 20000 40000 60000 100000 --samples 2 --warm --out "$R/needles-$1.jsonl" 2>&1 | tee -a "$R/gauntlet.log" | grep -E "SUMMARY|MISS"
  log "needles($1) rc=${PIPESTATUS[0]}"
}
seangate() { # $1=tag ; cold+cached multi-request 32K+ recall under 4 loaders, 30 samples
  log "--- sean gate ($1): 4 loaders x 20K, depths 32K/48K/64K x5, warm ---"
  python3 $P --url $URL --model $M --depths 32000 48000 64000 --samples 5 --loaders 4 --parallel 2 --warm --seed sean-$1 --out "$R/seangate-$1.jsonl" 2>&1 | tee -a "$R/gauntlet.log" | grep -E "SUMMARY|MISS"
  log "seangate($1) rc=${PIPESTATUS[0]}"
}
killer() { # 8 concurrent fresh-seed 24K prompts; all must return 200 + non-empty
  log "--- killer 8x24K burst ---"
  python3 - "$URL" "$M" "$R/killer.jsonl" <<'PY' 2>&1 | tee -a "$R/gauntlet.log"
import json,random,sys,time,urllib.request,concurrent.futures as cf
url,model,out=sys.argv[1:4]; W="alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa".split()
def one(i):
    rng=random.Random(f"killer-{time.time()}-{i}"); body=" ".join(rng.choice(W) for _ in range(int(24000/1.3)))
    req={"model":model,"max_tokens":400,"temperature":0.6,"messages":[{"role":"user","content":body+"\n\nReply with the single word OK."}]}  # 400: reasoning-on model, 64 starved content
    t=time.time()
    try:
        d=json.load(urllib.request.urlopen(urllib.request.Request(url+"/chat/completions",data=json.dumps(req).encode(),headers={"Content-Type":"application/json"}),timeout=900))
        m=d["choices"][0]["message"]; c=(m.get("content") or "").strip(); r=(m.get("reasoning") or m.get("reasoning_content") or "").strip()
        return {"i":i,"ok":bool(c or r),"s":round(time.time()-t,1),"ptok":d["usage"]["prompt_tokens"],"ctok":d["usage"]["completion_tokens"],"finish":d["choices"][0]["finish_reason"],"ans":(c or r)[:40]}
    except Exception as e: return {"i":i,"ok":False,"err":repr(e)[:120]}
with cf.ThreadPoolExecutor(8) as ex: res=list(ex.map(one,range(8)))
open(out,"a").write("\n".join(json.dumps(r) for r in res)+"\n")
ok=sum(r["ok"] for r in res); print(f"KILLER {ok}/8", [r.get("err") or r["ans"] for r in res if not r["ok"]] or "")
PY
}
vision() { # 8 concurrent x 4 2048^2 PNGs
  log "--- vision burst 8x4img ---"
  python3 - "$URL" "$M" <<'PY' 2>&1 | tee -a "$R/gauntlet.log"
import base64,json,struct,sys,zlib,urllib.request,concurrent.futures as cf
url,model=sys.argv[1:3]; W=H=2048
def chunk(t,d): return struct.pack('>I',len(d))+t+d+struct.pack('>I',zlib.crc32(t+d)&0xffffffff)
raw=bytearray()
for y in range(H):
    raw.append(0); raw+=bytes(v for x in range(W) for v in ((x*255)//W,(y*255)//H,((x+y)*255)//(W+H)))
png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',W,H,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(bytes(raw),6))+chunk(b'IEND',b'')
img={"type":"image_url","image_url":{"url":"data:image/png;base64,"+base64.b64encode(png).decode()}}
def one(j):
    body={"model":model,"max_tokens":400,"temperature":0.6,"messages":[{"role":"user","content":[{"type":"text","text":f"Stream {j}: describe the four images briefly."}]+[img]*4}]}  # 400: reasoning-on
    try:
        d=json.load(urllib.request.urlopen(urllib.request.Request(url+"/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"}),timeout=600)); m=d["choices"][0]["message"]; return bool((m.get("content") or "").strip() or (m.get("reasoning") or m.get("reasoning_content") or "").strip())
    except Exception as e: print("vision err",repr(e)[:100]); return False
with cf.ThreadPoolExecutor(8) as ex: r=list(ex.map(one,range(1,9)))
print(f"VISION {sum(r)}/8")
PY
}
so_probe() {
  log "--- structured-output probe (json_schema x4) ---"
  python3 - "$URL" "$M" <<'PY' 2>&1 | tee -a "$R/gauntlet.log"
import json,sys,urllib.request
url,model=sys.argv[1:3]; ok=0
schema={"type":"object","properties":{"city":{"type":"string"},"population":{"type":"integer"},"landmarks":{"type":"array","items":{"type":"string"}}},"required":["city","population","landmarks"],"additionalProperties":False}
for q in ("Paris","Tokyo","Lima","Oslo"):
    body={"model":model,"max_tokens":400,"temperature":0.6,"messages":[{"role":"user","content":f"Give facts about {q} as JSON."}],"response_format":{"type":"json_schema","json_schema":{"name":"facts","schema":schema,"strict":True}}}
    try:
        d=json.load(urllib.request.urlopen(urllib.request.Request(url+"/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"}),timeout=300))
        o=json.loads(d["choices"][0]["message"]["content"]); ok+= set(o)=={"city","population","landmarks"}
    except Exception as e: print("so err",repr(e)[:120])
print(f"SO {ok}/4")
PY
}
tooleval() {
  log "--- tool-eval 69x2 @T0.6 effort=medium (fp8 band: plain 91, tier-rc4 92.5) ---"
  cd "$HOME"  # tool-eval-bench writes ./data/benchmarks.sqlite + ./runs/ relative to cwd (systemd cwd is /)
  local TS; TS=$(date +%H%M)  # per-run files: a fixed name destroyed the overlay run's json once (R77)
  tool-eval-bench --base-url $URL --model $M --temperature 0.6 --top-p 0.95 --top-k 20 --trials 2 --parallel 8 --json-file "$R/tooleval-69x2-$TS.json" > "$R/tooleval-69x2-$TS.log" 2>&1
  grep -aiE "final_score|mean|Quality" "$R/tooleval-69x2-$TS.log" | tail -5 | tee -a "$R/gauntlet.log"
}
benchy() {
  log "--- benchy: ladder pp8192 c1/2/4/8, deep pp30K c1 AND c8, prefill 8K/32K/100K ---"
  B="llama-benchy --base-url $URL --model $M"
  { for c in 1 2 4 8; do $B --pp 8192 --tg 512 --concurrency $c --runs 2; done
    echo "=== deep pp30000 c1 tg512 (vs fp8; July FA2-nvfp4 was -8..-23% here) ==="; $B --pp 30000 --tg 512 --concurrency 1 --runs 2
    echo "=== deep pp30000 c8 tg512 ==="; $B --pp 30000 --tg 512 --concurrency 8 --runs 1
    for p in 8192 32768 100000; do $B --pp $p --tg 16 --concurrency 1 --runs 2; done
    echo "=== BENCHY-DONE ==="; } > "$R/benchy.log" 2>&1
  log "benchy done ($(grep -c . "$R/benchy.log") lines)"
}

diag() { # negative control: in-tree swizzled-V-scale writer + FA2 linear reader -> deep needles MUST miss
  boot 4 0 swizzled-diag || { log "DIAG boot with LINEAR_VSF=0 failed — see boot-swizzled-diag.log"; return 1; }
  needles swizzled-diag
  log "DIAG: LINEAR_VSF=0 needles rc above — MUST be non-zero (misses) or the layout theory is wrong"
}
cliff_metrics() { # engine-side view of the >=50K-tokens-in-flight cliff (NS=4): preemptions / spec counters / queue
  log "--- cliff metrics: /metrics snapshot around c8 pp8192 tg128 ---"
  snap() { curl -s http://localhost:8029/metrics | grep -aE "^vllm:(num_preemptions_total|num_requests_(running|waiting)|spec_decode_num_(drafts|accepted_tokens|draft_tokens)_total|kv_cache_usage_perc|prefix_cache_(hits|queries)_total|request_(prefill|decode|queue)_time_seconds_sum)[{ ]" ; }  # labels follow the name with '{'

  snap > "$R/metrics-before.txt"
  llama-benchy --base-url $URL --model $M --pp 8192 --tg 128 --concurrency 8 --runs 1 > "$R/benchy-cliff-metrics.log" 2>&1
  snap > "$R/metrics-after.txt"
  python3 - "$R/metrics-before.txt" "$R/metrics-after.txt" <<'PY2' | tee -a "$R/gauntlet.log"
import sys,re
def load(f):
    d={}
    for l in open(f):
        m=re.match(r"(\S+?)(\{[^}]*\})? (\S+)",l.strip())
        if m: d[m.group(1)+(m.group(2) or "")]=float(m.group(3))
    return d
a,b=load(sys.argv[1]),load(sys.argv[2])
for k in sorted(b):
    if "running" in k or "waiting" in k or "usage" in k: print(f"METRIC {k} = {b[k]}")
    elif b[k]-a.get(k,0)!=0: print(f"METRIC delta {k} = {b[k]-a.get(k,0):.3f}")
PY2
  sudo docker logs vllm-exp 2>&1 | grep -aiE "preempt|recompute|warn.*(kv|cache|spec)" | tail -5 | cut -c1-200 | tee -a "$R/gauntlet.log"
}
benchy_cliff() { # straddle the 65,536-total-token boundary (8x8192 cliff: 159 t/s vs 296 at 8x6144)
  log "--- benchy cliff: c8 pp8000 | c8 pp8256 | c7 pp8192 | c8 pp9216 | c4 pp16384 (=65,536 total) ---"
  B="llama-benchy --base-url $URL --model $M"
  { $B --pp 8000 --tg 128 --concurrency 8 --runs 2; $B --pp 8256 --tg 128 --concurrency 8 --runs 2; $B --pp 8192 --tg 128 --concurrency 7 --runs 2; $B --pp 9216 --tg 128 --concurrency 8 --runs 2; $B --pp 16384 --tg 128 --concurrency 4 --runs 2; echo "=== CLIFF-DONE ==="; } >> "$R/benchy-cliff-$(date +%H%M).log" 2>&1
  grep -ahE "Running test|tg128" "$R"/benchy-cliff-*.log | grep -av "^|:" | tail -10 | cut -c1-120 | tee -a "$R/gauntlet.log"
}
benchy_c8_matrix() { # bracket the 8x8K prefill cliff: concurrency x prompt length on the current engine
  log "--- benchy c8 matrix: pp8192 c4/c6, pp6144 c8, pp12288 c4 ---"
  B="llama-benchy --base-url $URL --model $M"
  { $B --pp 8192 --tg 256 --concurrency 4 --runs 2; $B --pp 8192 --tg 256 --concurrency 6 --runs 2; $B --pp 6144 --tg 256 --concurrency 8 --runs 2; $B --pp 12288 --tg 256 --concurrency 4 --runs 2; echo "=== MATRIX-DONE ==="; } >> "$R/benchy-c8-matrix-$(date +%H%M).log" 2>&1
  grep -ahE "Running test|tg256" "$R"/benchy-c8-matrix-*.log | grep -av "^|:" | tail -8 | cut -c1-120 | tee -a "$R/gauntlet.log"
}
benchy_c8() { # re-check of the c8 ladder point (first run: 162 t/s, TTFT 6.4s±7.5s — outlier vs 321.7 fp8)
  log "--- benchy c8 re-run: pp8192 c8 x3, pp4096 c8 ---"
  B="llama-benchy --base-url $URL --model $M"
  { $B --pp 8192 --tg 512 --concurrency 8 --runs 3; $B --pp 4096 --tg 512 --concurrency 8 --runs 2; echo "=== BENCHY-C8-DONE ==="; } > "$R/benchy-c8-redo.log" 2>&1
  grep -aE "tg512" "$R/benchy-c8-redo.log" | grep -av "^|:" | cut -c1-120 | tee -a "$R/gauntlet.log"
}

if [ -n "${STAGES:-}" ]; then  # partial re-run on a fresh NS=4 engine, e.g. STAGES="killer vision tooleval"
  log "=== NVFP4KV GAUNTLET partial: $STAGES ==="
  boot "${REDO_NS:-4}" "${REDO_VSF:-1}" "ns${REDO_NS:-4}-vsf${REDO_VSF:-1}-redo" || { log "BOOT redo FAILED"; exit 1; }  # REDO_NS=0 native path; REDO_VSF=0 swizzled diagnostic
  for st in $STAGES; do $st; done
  log "=== partial done; engine left up on :8029 ==="; exit 0
fi

log "=== NVFP4KV GAUNTLET start ==="
boot 4 1 ns4 || { log "BOOT ns4 FAILED — stop"; exit 1; }
needles ns4; seangate ns4; killer; vision; so_probe; tooleval; benchy
boot 0 1 ns0 || { log "BOOT ns0 FAILED"; }
needles ns0; seangate ns0
diag
sudo docker rm -f vllm-exp >/dev/null 2>&1
log "=== NVFP4KV GAUNTLET DONE — engine removed; restore the daily with launch-daily.sh ==="
