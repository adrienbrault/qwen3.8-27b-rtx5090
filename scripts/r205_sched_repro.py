#!/usr/bin/env python3
"""R205: drive the REAL Scheduler offline (CPU) on a hybrid 1 full-attn + 3 mamba_align layout and ask whether a second identical
prompt gets a prefix-cache hit, with and without the MTP speculative config. Usage (inside the served image, /model mounted):
    python3 r205_sched_repro.py [--hash 16] [--block 64] [--spec qwen3_5_mtp|none] [--ns 3] [--nblk 10] [--tail 7] [--drafts]
"""
import argparse
import sys

import torch

sys.path.insert(0, "/t")
from test_r205_repro import _make_hybrid_kv_cache_config, make_request  # noqa: E402

from vllm.config import (  # noqa: E402
    CacheConfig,
    KVTransferConfig,
    ModelConfig,
    ObservabilityConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.utils.hashing import sha256  # noqa: E402
from vllm.v1.core.kv_cache_utils import init_none_hash  # noqa: E402
from vllm.v1.core.sched.scheduler import Scheduler  # noqa: E402
from vllm.v1.outputs import ModelRunnerOutput  # noqa: E402
from vllm.v1.structured_output import StructuredOutputManager  # noqa: E402

try:
    from vllm.v1.core.sched.output import DraftTokenIds  # type: ignore
except Exception:  # pragma: no cover
    DraftTokenIds = None


def build(args):
    model_config = ModelConfig(model="/model", trust_remote_code=True, dtype="bfloat16", seed=42,
                               max_model_len=args.max_len)
    scheduler_config = SchedulerConfig(max_num_seqs=16, max_num_batched_tokens=args.chunk, max_model_len=args.max_len,
                                       enable_chunked_prefill=True, watermark=0.0, is_encoder_decoder=False)
    cache_config = CacheConfig(block_size=args.block, gpu_memory_utilization=0.9, cache_dtype="auto",
                               enable_prefix_caching=True, mamba_cache_mode="align",
                               **({"prefix_match_unit": args.hash} if args.hash != args.block else {}))
    spec = None
    if args.spec != "none":
        spec = SpeculativeConfig(method=args.spec, num_speculative_tokens=args.ns, target_model_config=model_config,
                                 target_parallel_config=ParallelConfig())
    ktc = None
    if args.offload:
        ktc = KVTransferConfig(kv_connector="OffloadingConnector", kv_role="kv_both", kv_connector_extra_config={
            "spec_name": "TieringOffloadingSpec", "cpu_bytes_to_use": 17179869184, "offload_prompt_only": True,
            "secondary_tiers": [{"type": "fs", "root_dir": "/l2", "n_read_threads": 16, "n_write_threads": 4}]})
    vllm_config = VllmConfig(scheduler_config=scheduler_config, model_config=model_config, cache_config=cache_config,
                             parallel_config=ParallelConfig(), speculative_config=spec, kv_transfer_config=ktc,
                             observability_config=ObservabilityConfig())
    order = ["full", "mamba_align", "mamba_align", "mamba_align"] if args.order == "full-first" else ["mamba_align", "mamba_align", "mamba_align", "full"]
    kv = _make_hybrid_kv_cache_config(args.block, 4000, order)
    if spec is not None and args.spec_blocks:
        from dataclasses import replace
        from vllm.v1.kv_cache_interface import KVCacheGroupSpec, MambaSpec
        groups = []
        for g in kv.kv_cache_groups:
            s = g.kv_cache_spec
            if isinstance(s, MambaSpec):
                s = replace(s, num_speculative_blocks=args.ns)
            groups.append(KVCacheGroupSpec(g.layer_names, s))
        kv = replace(kv, kv_cache_groups=groups)
    if args.retention != "none":
        from dataclasses import replace
        kv = replace(kv, prefix_cache_retention_interval=None if args.retention == "dense" else int(args.retention))
    cache_config.num_gpu_blocks = 4000
    sched = Scheduler(vllm_config=vllm_config, kv_cache_config=kv, block_size=args.block, hash_block_size=args.hash,
                      log_stats=True, structured_output_manager=StructuredOutputManager(vllm_config))
    sched.use_v2_model_runner = True
    sc = vllm_config.speculative_config
    print(f"scheduler: retention={sched.kv_cache_manager.coordinator.retention_interval} use_eagle={sched.use_eagle} kv_block_drop={sched.use_eagle_kv_block_drop} lookahead={sched.num_prefill_lookahead} "
          f"need_split={sched.need_mamba_block_aligned_split} partial_hit={sched.mamba_partial_cache_hit} hash={sched.hash_block_size} "
          f"ckpt_blocks={sched.mamba_has_prefill_checkpoint_blocks} partial_hash={sched.kv_cache_manager.coordinator.enable_partial_hash_hits} eagle_groups={sorted(sched.kv_cache_manager.coordinator.eagle_group_ids)} spec={None if sc is None else (sc.method, sc.num_speculative_tokens)}")
    return sched


def run_request(sched, req, args, drafts):
    sched.add_request(req)
    steps = 0
    while not req.is_finished() and steps < 400:
        out = sched.schedule()
        ids = [r.req_id for r in out.scheduled_new_reqs] + list(out.scheduled_cached_reqs.req_ids)
        sampled = []
        for rid in ids:
            r = sched.requests[rid]
            n = out.num_scheduled_tokens[rid]
            done_prompt = r.num_computed_tokens + n >= r.num_prompt_tokens
            if not done_prompt:
                sampled.append([])
            else:
                # accept every scheduled spec token plus one new token (keeps the MTP path realistic)
                nspec = len(out.scheduled_spec_decode_tokens.get(rid, [])) if getattr(out, "scheduled_spec_decode_tokens", None) else 0
                sampled.append([1000 + steps] * (nspec + 1))
        mro = ModelRunnerOutput(req_ids=ids, req_id_to_index={r: i for i, r in enumerate(ids)}, sampled_token_ids=sampled,
                                logprobs=None, prompt_logprobs_dict={}, pooler_output=[None] * len(ids), kv_connector_output=None)
        sched.update_from_output(out, mro)
        if drafts and DraftTokenIds is not None:
            live = [rid for rid in ids if rid in sched.requests and not sched.requests[rid].is_finished()
                    and sched.requests[rid].num_computed_tokens >= sched.requests[rid].num_prompt_tokens]
            if live:
                sched.update_draft_token_ids(DraftTokenIds(live, [[7, 7, 7][: args.ns] for _ in live]))
        steps += 1
        if steps <= 6 or done_prompt and steps < 12:
            pass
    return steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hash", type=int, default=16)
    p.add_argument("--block", type=int, default=64)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--spec", default="qwen3_5_mtp")
    p.add_argument("--ns", type=int, default=3)
    p.add_argument("--nblk", type=int, default=10)
    p.add_argument("--tail", type=int, default=7)
    p.add_argument("--drafts", action="store_true")
    p.add_argument("--spec-blocks", action="store_true")
    p.add_argument("--order", default="full-first")
    p.add_argument("--max-len", type=int, default=8192)
    p.add_argument("--offload", action="store_true")
    p.add_argument("--retention", default="none", help="none = leave the synthetic config (dense); dense; or an int (live default 0)")
    args = p.parse_args()
    init_none_hash(sha256)
    sched = build(args)
    tokens = [i for i in range(args.nblk) for _ in range(args.block)] + [args.nblk] * args.tail
    req0 = make_request("0", tokens, args.hash, sha256)
    steps = run_request(sched, req0, args, args.drafts)
    st = sched.kv_cache_manager.make_prefix_cache_stats()
    pool = sched.kv_cache_manager.block_pool
    cached = {g: [i for i in range(len(req0.block_hashes)) if pool.get_cached_block(req0.block_hashes[i], kv_cache_group_ids=[g])] for g in range(4)}
    print(f"req0: steps={steps} finished={req0.is_finished()} stats(q,h)={(st.queries, st.hits) if st else None} cached_hashes_per_group={ {g: (len(v), v[-3:]) for g, v in cached.items()} }")
    req1 = make_request("1", tokens, args.hash, sha256)
    sched.add_request(req1)
    out = sched.schedule()
    st = sched.kv_cache_manager.make_prefix_cache_stats()
    n = out.num_scheduled_tokens.get("1")
    print(f"connector={type(sched.connector).__name__ if sched.connector else None} req0_finished_blocks_freed={req0.request_id not in sched.requests if hasattr(req0, 'request_id') else None}")
    print(f"RESULT spec={args.spec} hash={args.hash} block={args.block} chunk={args.chunk} order={args.order} offload={args.offload} drafts={args.drafts} retention={args.retention}: req1 prompt={len(tokens)} scheduled_first_chunk={n} "
          f"num_computed_tokens={req1.num_computed_tokens} num_cached={getattr(req1, 'num_cached_tokens', None)} stats(q,h)={(st.queries, st.hits) if st else None}")


if __name__ == "__main__":
    main()
