# Tier stack for vLLM nightly 0.27 + LMCache v0.5.4rc4 (2026-08-15)

Codex audit (AUDIT.md) + local rebase. Verdicts vs the 0.23/0.5.2 stack:
- 0003 RETIRED — fixed upstream (fixed-point hybrid hit reduction in vLLM 0.27).
- 0001/0002 RE-DERIVED for 0.27's layout contract (fused rank-4 view + strided fp8
  regroup) — MUST pass the runtime needle gate before trusting; static analysis could
  not prove the production stride.
- 0005/0007/0008 rebased; 0009 NEW (fs_native watermark LRU eviction — the cap was
  enforce-only before, L2 froze read-only when full); 0010 (xfer-abort assert) baked in.
- Ordered stack applies to the v0.5.4rc4 tag: 0001→0002→0007→0008→0009 (one cosmetic
  comment-hunk in 0009/connector.cpp dropped vs rc4's O_DIRECT refactor).
  rc4-stack.diff = the whole applied stack as one diff. Dockerfile.rc4 builds
  vllm-qwen38:tiers-rc4 with CPU build-gate tests.
