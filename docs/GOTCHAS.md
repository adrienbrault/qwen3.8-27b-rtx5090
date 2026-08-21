# Gotchas

Flag rationale is in [CONFIG.md](CONFIG.md); this page is the failure modes, newest first.

1. **Chunk = unified block.** vLLM sizes the hybrid's attention block to cover the Mamba page: 1616 tokens with fp8 KV, 2864 with nvfp4 (logged at boot). The LMCache chunk and `mnbt = 2·chunk − 1` follow it. Changing the KV dtype without re-deriving them mis-chunks the cache.
2. **Verify the engine, not the launcher banner.** After every launch check the container image, `vllm.__version__`, `Using V2 Model Runner`, `NVFP4KV-SM120: linear-V-scale store overlay ACTIVE` (and no `swizzled-V-scale writer` line), and the pool in the band. A stale `export IMAGE=` in a dispatcher ran a 0.23 image under a "tier-rc4" banner for six days in 2026-08; the pool number coincided.
3. **The nvfp4 overlay is required and invisible to behavioural tests.** With the in-tree swizzled writer, needles to 100K and a 69×2 tool-eval pass; the numeric diagnostic shows 2.7–10× the attention error. Run [`../patches-nvfp4kv/overlay/diag_vsf_layout.py`](../patches-nvfp4kv/overlay/diag_vsf_layout.py) after any vLLM bump that touches `csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu`.
4. **Util on the V2 runner.** 0.95 with fp8 KV; 0.93 with nvfp4 KV (at 0.95 the FlashInfer autotuner OOM-falls-back on the first new shape); 0.98 on the plain profile OOMs inside the `fp4_gemm` autotuner on the first request.
5. **A flaky spawn-child ImportError on nightly `ba07e4a48`.** `_C_stable_libtorch.abi3.so: undefined symbol: …cutlass…GemmUniversalBase…` in the engine-core child, once in six boots. A retry boots clean; the dispatcher's health loop covers it.
6. **Pool varies ±6% boot to boot** on this nightly (nvfp4 NS=0: 443K then 415K; fp8 tier: 209,859 then 208,450). Use bands, not exact values.
7. **Fresh L2 namespace per stack generation.** The on-disk page format changes with the LMCache build and the KV dtype. Poisoned chunks are not repaired.
8. **`--no-async-scheduling` with MTP.** About 1 tool-eval point on this hybrid, and the historical cause of KV corruption under spec decode ([vllm#42655](https://github.com/vllm-project/vllm/issues/42655)).
9. **`--mamba-cache-mode align` is load-bearing.** About 3 tool-eval points with spec decode, and what gives LMCache a scheduler-sized page.
10. **Never `PYTORCH_ALLOC_CONF=expandable_segments` with the connector.** `register_kv_caches` times out after 300 s with no useful error.
11. **Reasoning models starve `content` on small `max_tokens`.** A 64-token probe returns empty content because the budget goes to reasoning. Score content-or-reasoning and budget 400+ in probes.
12. **The tool parser stays on.** A harness that sends `tools` without `tool_choice` gets HTTP 400 without `--enable-auto-tool-choice --tool-call-parser qwen3_xml`. The qwen3_xml parser does strip in-text `<function=>` XML from tool-less `tool_choice=none` requests; that only bites when a harness has function calling off, which is a harness bug.
13. **Harness function-calling gates.** R2E-Gym's local patch enabled OpenAI function calling only for model names matching a pattern; serving under a new name silently turned it off (step-0 prompt 943 vs 2,519 tokens, tool names hallucinated, 5/25 instead of 20/25). Check `Using fn calling: True` and the step-0 prompt size before trusting an agentic run.
14. **Check `tokenizer.json` for a baked truncation.** The W4A4 export shipped `"truncation": {"max_length": 8192}` from calibration: text works, multimodal requests past 8192 tokens fail with HTTP 400. The launcher nulls it at every boot; a re-download reinstates it.
15. **`max_capacity_gb` on `fs_native` is telemetry without patches 0008/0009.** `du -sh` the L2 directory for the first day of any new generation.
16. **The sidecar's VRAM is invisible to `--gpu-memory-utilization`** (796 MiB). Port the profile and it OOMs at a utilization that "worked" on the plain engine.
17. **vLLM's `prefix_cache_hits_total` reads 0 with the connector on** — the connector owns the lookups. Time a repeated prompt instead ([`../bench/prefix_probe.py`](../bench/prefix_probe.py)).
18. **Needle-test across a restart before trusting any external KV tier on a hybrid model.** Coherent output and rising hit counters were compatible with a broken cache through four rounds ([LMCACHE.md](LMCACHE.md)).
19. **Self-matching `pkill -f`, and procps `kill -SIG -<pgid>`.** The latter keeps the first digit after the dash and sends `kill(-1)`; that took the box offline for five days in 2026-08. Use the bash builtin with `--`, or systemd units.
