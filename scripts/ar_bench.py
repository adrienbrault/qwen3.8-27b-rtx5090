#!/usr/bin/env python3
"""R184: all-reduce microbenchmark at the decode shapes of the two-card daily.

torchrun --nproc_per_node=2 ar_bench.py --hidden 5120 --rows 1,8,10,16,20,40,80,160,320,2048,8192

Backends (each rank times its own calls; the table reports the group max of the
per-rank medians, as flashinfer's own comm benchmark does):
  nccl  torch.distributed.all_reduce on the NCCL group (vLLM's PYNCCL fallback)
  vllm  vLLM's CustomAllreduce (the served daily's decode path; max_size raised to
        256 MiB so the prefill-sized rows go through it too)
  pcie  flashinfer.comm.PcieIpcAllReduceWorkspace (flashinfer main, PR #4393)
  vend  pcie_ipc_ar21.workspace.PcieIpcAllReduceWorkspace: the same kernel vendored into the served
        image by patch 0138 (R185) with the fixed R184 tune table; rows <= 320 only (its slab is sized
        for the decode shapes). Needs an image that carries the package.

Row = tokens in the step (c1 with a 9-token draft = 10 rows; c8 = 80; c16 = 160;
a 2,048 / 8,192-token prefill chunk = 2048 / 8192). Both eager and CUDA-graph
replay are timed; the daily runs decode inside graphs.
"""
import argparse, contextlib, json, os, statistics, sys, time
import torch, torch.distributed as dist


def log(rank, *a):
    if rank == 0:
        print(*a, flush=True)


def group_max(v, device, group):
    t = torch.tensor([v], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    return float(t.item())


def time_eager(fn, iters, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record(); torch.cuda.synchronize()
        samples.append(s.elapsed_time(e) * 1000.0 / iters)
    return statistics.median(samples)


def time_graph(fn, iters, capture_ctx=None, replays=10):
    # the eager pass already warmed every kernel; warm on the CURRENT stream (the pcie_ipc workspace
    # is bound to one stream and rejects a side-stream call), then capture `iters` calls
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    import contextlib
    ctx = capture_ctx() if capture_ctx else contextlib.nullcontext()
    with ctx:
        with torch.cuda.graph(g):
            for _ in range(iters):
                fn()
    torch.cuda.synchronize()
    g.replay(); torch.cuda.synchronize()
    samples = []
    for _ in range(replays):
        st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
        st.record(); g.replay(); en.record(); torch.cuda.synchronize()
        samples.append(st.elapsed_time(en) * 1000.0 / iters)
    return statistics.median(samples), g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--rows", default="1,8,10,16,20,40,80,160,320,2048,8192")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--graph-iters", type=int, default=50)
    ap.add_argument("--backends", default="nccl,vllm,pcie")
    ap.add_argument("--json", default="")
    ap.add_argument("--tune-cache", default="")
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--jit-only", action="store_true", help="build the pcie_ipc module and exit")
    a = ap.parse_args()

    local = int(os.environ.get("LOCAL_RANK", "0")); torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    rank = dist.get_rank(); world = dist.get_world_size()
    device = torch.device("cuda", rank)
    group = dist.group.WORLD
    rows = [int(x) for x in a.rows.split(",")]
    H = a.hidden; dtype = torch.bfloat16
    backends = a.backends.split(",")
    res = {"hidden": H, "world": world, "rows": rows, "eager": {}, "graph": {}, "notes": {}}

    pcie = None
    if "pcie" in backends:
        import flashinfer, flashinfer.comm as comm
        log(rank, f"flashinfer {flashinfer.__version__} from {flashinfer.__file__}")
        t0 = time.time()
        comm.pcie_ipc_ar.get_pcie_ipc_comm_module()
        log(rank, f"pcie_ipc module ready in {time.time()-t0:.0f}s")
        if a.jit_only:
            dist.barrier(); dist.destroy_process_group(); return
        kw = {}
        if a.tune_cache:
            kw["tune_cache"] = a.tune_cache
        pcie = comm.PcieIpcAllReduceWorkspace(group=group, max_numel=max(rows) * H, dtype=dtype, **kw)
        if not a.no_tune:
            t0 = time.time()
            pcie.tune([H], dtype=dtype, cache=a.tune_cache or None)
            log(rank, f"pcie_ipc tuned in {time.time()-t0:.0f}s")
        pcie.prepare([(r, H) for r in rows], dtype=dtype)
        res["notes"]["pcie_supports"] = {r: bool(pcie.supports(torch.empty(r, H, dtype=dtype, device=device))) for r in rows}
        log(rank, "pcie supports:", res["notes"]["pcie_supports"])

    ca = None; vend = None
    # vLLM attaches the custom all-reduce (and the vendored pcie_ipc workspace) to a CPU (gloo) group for its IPC-handle exchange
    gloo = dist.new_group(backend="gloo") if ("vllm" in backends or "vend" in backends) else None
    if "vllm" in backends:
        from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
        ca = CustomAllreduce(group=gloo, device=device, max_size=256 << 20)
        res["notes"]["vllm_disabled"] = bool(ca.disabled)
        log(rank, "vllm CustomAllreduce disabled:", ca.disabled, "full_nvlink:", getattr(ca, "full_nvlink", None))
    if "vend" in backends:
        import pcie_ipc_ar21
        from pcie_ipc_ar21.workspace import PcieIpcAllReduceWorkspace as VendWorkspace
        log(rank, f"pcie_ipc_ar21 from {pcie_ipc_ar21.__file__}")
        vend = VendWorkspace(gloo, device)
        added = vend.prepare([(r, H) for r in rows if r <= vend.max_numel // H], dtype=dtype)
        res["notes"]["vend_configs"] = {f"{r}x{h}": list(c) for (r, h), c in added.items()}
        res["notes"]["vend_supports"] = {r: bool(vend.supports(torch.empty(r, H, dtype=dtype, device=device))) for r in rows}
        log(rank, "vend configs (blocks,threads,variant):", res["notes"]["vend_configs"])

    def make_fn(name, x):
        if name == "nccl":
            return lambda: dist.all_reduce(x, group=group)
        if name == "vllm":
            if ca is None or ca.disabled or not ca.should_custom_ar(x):
                return None
            return lambda: ca.custom_all_reduce(x)
        if name == "pcie":
            if pcie is None or not pcie.supports(x):
                return None
            return lambda: pcie.all_reduce(x)
        if name == "vend":
            if vend is None or not vend.supports(x) or not vend.is_resolved(x):
                return None
            return lambda: vend.all_reduce(x)
        raise ValueError(name)

    def capture_ctx(name):
        if name == "vllm":
            return ca.capture
        if name == "vend":
            return vend.capture
        return None

    hdr = f"{'rows':>6} {'bytes':>10} | " + " ".join(f"{b+'-eager':>11} {b+'-graph':>11}" for b in backends)
    log(rank, hdr)
    def dump():
        if rank == 0 and a.json:
            with open(a.json, "w") as f:
                json.dump(res, f, indent=1)

    for r in rows:
        # collective skip when the row does not fit next to whatever else holds the card (~6 tensors + slack)
        need = 6 * r * H * 2 + (64 << 20)
        free = torch.cuda.mem_get_info(device)[0]
        free = -group_max(-free, device, group)
        if free < need:
            log(rank, f"{r:>6} {r*H*2:>10} | skipped: {free/2**20:.0f} MiB free on the tightest card, need {need/2**20:.0f}")
            res["notes"].setdefault("skipped_rows", []).append([r, int(free)])
            continue
        torch.manual_seed(1234 + rank)
        x0 = torch.randn(r, H, dtype=dtype, device=device)
        # reference via NCCL for the correctness check
        ref = x0.clone(); dist.all_reduce(ref, group=group)
        line = f"{r:>6} {r*H*2:>10} | "
        res["eager"][r] = {}; res["graph"][r] = {}
        for b in backends:
          try:
              x = x0.clone()
              fn = make_fn(b, x)
              if fn is None:
                  res["eager"][r][b] = None; res["graph"][r][b] = None
                  line += f"{'n/a':>11} {'n/a':>11} "
                  continue
              # correctness (one call, fresh input)
              xc = x0.clone()
              out = make_fn(b, xc)()
              got = out if isinstance(out, torch.Tensor) else xc
              ok = torch.allclose(got.float(), ref.float(), rtol=2e-2, atol=2e-2)
              if not ok:
                  res["notes"].setdefault("mismatch", []).append([b, r, float((got.float() - ref.float()).abs().max())])
              dist.barrier()
              # eager: NCCL is in-place, so re-clone each iteration would add a copy; accept the
              # accumulating values (bf16 overflow to inf is fine for timing) — same for all backends
              x = x0.clone(); fn = make_fn(b, x)
              te = time_eager(fn, a.iters)
              te = group_max(te, device, group)
              dist.barrier()
              x = x0.clone(); fn = make_fn(b, x)
              try:
                  tg, g = time_graph(fn, a.graph_iters, capture_ctx=capture_ctx(b))
                  tg = group_max(tg, device, group)
                  del g
                  # graph-replayed correctness: one captured call from a fresh copy of x0 into a fixed buffer,
                  # replayed three times (a broken protocol shows on repetition, not on the first call)
                  xv = x0.clone(); buf = torch.empty_like(x0); fnv = make_fn(b, xv)
                  def one():
                      xv.copy_(x0); r = fnv(); buf.copy_(r if isinstance(r, torch.Tensor) else xv)
                  for _ in range(2):
                      one()
                  torch.cuda.synchronize()
                  gv = torch.cuda.CUDAGraph()
                  cctx = capture_ctx(b); ctxv = cctx() if cctx else contextlib.nullcontext()
                  with ctxv:
                      with torch.cuda.graph(gv):
                          one()
                  torch.cuda.synchronize()
                  for k in range(3):
                      gv.replay(); torch.cuda.synchronize()
                      err = float((buf.float() - ref.float()).abs().max())
                      err = group_max(err, device, group)
                      if not torch.allclose(buf.float(), ref.float(), rtol=2e-2, atol=2e-2):
                          res["notes"].setdefault("graph_mismatch", []).append([b, r, k, err])
                          log(rank, f"  GRAPH REPLAY MISMATCH {b} rows={r} replay={k} max|err|={err:.4g}")
                  res["notes"].setdefault("graph_maxerr", {}).setdefault(b, {})[r] = err
                  del gv
              except Exception as e:  # noqa
                  res["notes"].setdefault("graph_fail", []).append([b, r, repr(e)[:300]])
                  log(rank, f"  graph capture failed for {b} rows={r}: {repr(e)[:300]}")
                  tg = None
              dist.barrier()
              res["eager"][r][b] = te; res["graph"][r][b] = tg
              line += f"{te:>9.1f}us {(f'{tg:.1f}us' if tg is not None else 'fail'):>11} "
          except torch.OutOfMemoryError as e:
            res["notes"].setdefault("oom", []).append([b, r, repr(e)[:200]])
            log(rank, f"  OOM for {b} rows={r} (engine holding the card?) — skipping")
            res["eager"][r][b] = None; res["graph"][r][b] = None
            line += f"{'oom':>11} {'oom':>11} "
            torch.cuda.empty_cache()

        log(rank, line)
        dump()

    if pcie is not None:
        pcie.destroy()
    if vend is not None:
        vend.destroy()
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
