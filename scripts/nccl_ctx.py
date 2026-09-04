#!/usr/bin/env python3
"""For each ncclDevKernel_AllReduce kernel in a torch trace: duration + the kernel names just before and after it on the GPU timeline."""
import gzip, json, sys, collections
p = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
d = json.load(gzip.open(p, "rt")); ev = d.get("traceEvents", d)
k = sorted([e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")], key=lambda e: e["ts"])
idx = [i for i, e in enumerate(k) if "ncclDevKernel_AllReduce" in e["name"]]
print(f"kernels={len(k)} nccl_allreduce={len(idx)}")
durs = [k[i]["dur"] for i in idx]; print("nccl dur us: min %.0f median %.0f max %.0f" % (min(durs), sorted(durs)[len(durs)//2], max(durs)))
prev = collections.Counter(); nxt = collections.Counter()
for i in idx:
    prev[k[i-1]["name"][:70]] += 1; nxt[k[i+1]["name"][:70]] += 1
print("-- kernel BEFORE nccl allreduce:"); [print(f"  {c:5d} {nm}") for nm, c in prev.most_common(6)]
print("-- kernel AFTER nccl allreduce:"); [print(f"  {c:5d} {nm}") for nm, c in nxt.most_common(6)]
print("-- sample sequences (8 kernels before -> nccl):")
for i in idx[:n:max(1, len(idx)//n)]:
    print("  ", " | ".join(k[j]["name"][:28] for j in range(max(0, i-8), i)), "=> nccl %.0fus" % k[i]["dur"])
