#!/usr/bin/env python3
"""Patch 4 (v2): relax the spec-verify routing guard (PR #40914 review finding).

The inner guard `query_start_loc.shape[0] == B + 1` fails on CUDA-graph
captured steps because qsl is PADDED to max batch size (@rmarnold on the PR).
The silent fallback routes spec-verify batches into _prefill_attention, whose
large-continuation branch is the original #40880 capture-poisoned path ->
stale attention output -> degenerate "!!!!" with 0% draft acceptance.
Seen in production on concurrent long-context sessions (opencode subagents).

v2 uses `>=` (NOT a plain drop — v1's `if True:` also corrupted unpadded
paths): B derives from num_actual_tokens which excludes padding, so real
rows compute correctly and padded tail rows are discarded by the caller,
the same contract vLLM's own decode kernels rely on under capture.
"""
path = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/turboquant_attn.py"
src = open(path).read()
old = "            if attn_metadata.query_start_loc.shape[0] == B + 1:"
new = ("            # PATCH4v2: >= not == (qsl is padded under cudagraph capture;\n"
       "            # == silently fell back to the buggy continuation-prefill path).\n"
       "            if attn_metadata.query_start_loc.shape[0] >= B + 1:")
assert src.count(old) == 1, f"guard anchor found {src.count(old)}x"
open(path, "w").write(src.replace(old, new))
import ast
ast.parse(open(path).read())
print("PATCH4_OK")
