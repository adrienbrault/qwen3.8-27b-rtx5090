#!/usr/bin/env bash
# Unattended soak of the R81 daily (tiers + nvfp4 KV + V2) on :8020 — the thing to see is the FIRST
# L2 EVICTION on nvfp4 pages (fs_native cap 200 GB, trigger 0.8 → 160 GB; rc4 0008/0009 unexercised at
# this page size) and correct recall AFTER eviction. Cycle (every ~30 s): 4 parallel fresh 20–50K
# prompts (stores), 1 revisit of a prompt from 20 cycles ago (retrieve, possibly post-eviction), every
# 10 cycles a depth needle (exact-secret recall, fresh seed) and a revisit-needle from cycle-40; log VRAM,
# L2 size, eviction events, preemptions, health. Stops on: needle MISS, health down, disk < 60 GB.
#   sudo systemd-run --unit=soak-nvfp4 --collect -p User=adrienbrault -E HOURS=3 bash /srv/qwen5090/soak-nvfp4-l2.sh
set -uo pipefail
HOURS=${HOURS:-3}; U=http://localhost:8020/v1; M=qwen3.8-27b
R=/srv/qwen5090/results/2026-08-21-soak-nvfp4-l2; mkdir -p "$R"; L2=/srv/qwen5090/lmcache-l2-nvfp4-v2
log(){ echo "$(date -Is) $*" | tee -a "$R/soak.log"; }
evict_count(){ sudo docker exec vllm-27b sh -c 'grep -aE "L2 .*above watermark|fs_native.*(evicted|Evicted|removed)|L2EvictionController.*(evict|Evict)" /tmp/lmcache-server.log 2>/dev/null | grep -vE "trigger=0.800|Starting L2EvictionController" | wc -l' 2>/dev/null || echo 0; }
preempt(){ curl -s http://localhost:8020/metrics | grep -aE "^vllm:num_preemptions_total[{ ]" | awk '{print int($2)}'; }
vram(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
log "=== SOAK start HOURS=$HOURS L2=$(du -sh $L2 | cut -f1) vram=$(vram)MiB evict=$(evict_count) preempt=$(preempt) ==="
END=$(( $(date +%s) + HOURS*3600 )); cyc=0; ev0=$(evict_count); first_evict=""
python3 - "$U" "$M" "$R" "$END" <<'PY' 2>&1 | tee -a "$R/soak-traffic.log" &
import json,random,sys,time,urllib.request,concurrent.futures as cf
url,model,R,END=sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4])
W="alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo sierra tango".split()
hist=[]
def prompt(seed,n):
    rng=random.Random(seed); return " ".join(rng.choice(W) for _ in range(int(n/1.3)))+"\n\nSummarize the above in one sentence."
def ask(text,mt=120):
    t=time.time()
    try:
        d=json.load(urllib.request.urlopen(urllib.request.Request(url+"/chat/completions",data=json.dumps({"model":model,"max_tokens":mt,"temperature":0.6,"messages":[{"role":"user","content":text}]}).encode(),headers={"Content-Type":"application/json"}),timeout=900))
        return round(time.time()-t,1), d["usage"]["prompt_tokens"], None
    except Exception as e: return round(time.time()-t,1), None, repr(e)[:120]
cyc=0
while time.time()<END:
    cyc+=1; seeds=[(f"soak-{cyc}-{i}", random.Random(f"n-{cyc}-{i}").randint(20000,50000)) for i in range(4)]
    with cf.ThreadPoolExecutor(5) as ex:
        futs=[ex.submit(ask,prompt(s,n)) for s,n in seeds]
        rev=None
        if len(hist)>=20: rs,rn=hist[-20]; rev=ex.submit(ask,prompt(rs,rn))
        res=[f.result() for f in futs]; rr=rev.result() if rev else None
    hist.extend(seeds)
    errs=[r[2] for r in res if r[2]]
    print(json.dumps({"cycle":cyc,"t":time.strftime("%H:%M:%S"),"store_s":[r[0] for r in res],"ptok":[r[1] for r in res],"revisit_s":(rr[0] if rr else None),"revisit_err":(rr[2] if rr else None),"errors":errs}),flush=True)
    if errs: time.sleep(20)
PY
TRAFFIC=$!
while [ "$(date +%s)" -lt "$END" ]; do
  sleep 300; cyc=$((cyc+1))
  ev=$(evict_count); l2=$(du -sb $L2 | cut -f1); l2g=$((l2/1000000000)); free=$(df -BG /srv | tail -1 | awk '{print $4}' | tr -d G)
  up=$(curl -sf -m 5 http://localhost:8020/health >/dev/null && echo up || echo DOWN)
  log "tick $cyc: L2=${l2g}GB free=${free}G vram=$(vram)MiB evict_events=$ev preempt=$(preempt) daily=$up traffic_cycles=$(wc -l < $R/soak-traffic.log) traffic_errors=$(grep -c '"errors": \[\"' $R/soak-traffic.log)"
  if [ "$ev" -gt "$ev0" ] && [ -z "$first_evict" ]; then first_evict=$cyc; log "### FIRST L2 EVICTION EVENT at L2=${l2g}GB ###"; sudo docker exec vllm-27b sh -c 'grep -aE "above watermark|evict|Evict" /tmp/lmcache-server.log | grep -v "trigger=0.800\|Starting\|L1 memory" | tail -3' 2>/dev/null | cut -c1-200 | tee -a "$R/soak.log"; fi
  if [ "$up" != up ]; then log "### DAILY DOWN — stopping soak ###"; break; fi
  if [ "$free" -lt 60 ]; then log "### DISK < 60 GB — stopping soak (cap not holding?) ###"; break; fi
  if [ $((cyc % 2)) = 0 ]; then   # every 10 min: fresh needle + a revisit needle (seed from 4 ticks ago) — exact recall, incl. post-eviction
    python3 /srv/qwen5090/probes/needle_depth.py --url $U --model $M --depths 40000 --samples 1 --seed "soak-$cyc" --out "$R/needles.jsonl" 2>&1 | grep -E "HIT|MISS" | cut -c1-120 | tee -a "$R/soak.log"
    if [ $cyc -ge 6 ]; then python3 /srv/qwen5090/probes/needle_depth.py --url $U --model $M --depths 40000 --samples 1 --seed "soak-$((cyc-4))" --out "$R/needles-revisit.jsonl" 2>&1 | grep -E "HIT|MISS" | sed 's/^/  revisit /' | cut -c1-120 | tee -a "$R/soak.log"; fi
    if tail -2 "$R/soak.log" | grep -q "MISS"; then log "### NEEDLE MISS — stopping soak, engine left up for forensics ###"; break; fi
  fi
done
kill $TRAFFIC 2>/dev/null; wait $TRAFFIC 2>/dev/null
log "=== SOAK END: L2=$(du -sh $L2 | cut -f1) vram=$(vram)MiB evict_events=$(evict_count) (first at tick ${first_evict:-never}) preempt=$(preempt) errors=$(grep -c '"errors": \[\"' $R/soak-traffic.log) needles=$(grep -c HIT $R/soak.log)/$(grep -cE 'HIT|MISS' $R/soak.log) ==="
echo SOAK-DONE >> "$R/soak.log"
