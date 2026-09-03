# Configuration of the served engine

Every flag of the two-card configuration, why it is set, and what happens without it. The launchers are [`scripts/serve-r156-daily.sh`](../scripts/serve-r156-daily.sh) (sets the two-card values and calls the generic launcher) and [`scripts/serve-v0280-daily.sh`](../scripts/serve-v0280-daily.sh) (the generic launcher, whose defaults are the one-card configuration). Inline comments in the scripts are the reference when the two disagree. Earlier generations are in [HISTORY.md](HISTORY.md) and [archive/](archive/).

## Image

`vllm-qwen38:v0280-nvfp4kv` = `vllm/vllm-openai:v0.28.0` plus [`patches-v0280/`](../patches-v0280/README-sm120-nvfp4.md). The two-card configuration uses from it: the DFlash2 quantized-drafter loader (0107), the DFlash2 selector sampling guard (0106), the GDN kernel hardening (0108) and the speculator CUDA graphs (0113). The one-card configuration additionally needs the NVFP4 KV cache pieces: FA2 routing (0101), the linear V-scale store overlay (0102), XQA decode (0103) and drafter graphs (0104). No GPU is needed to build the image.

## Container

```bash
--runtime nvidia --gpus all --ipc=host --shm-size 8g
--memory 52g --memory-swap 52g --oom-score-adj -800
-p ${BIND_ADDR}:${PORT}:8000
```

- `--ipc=host` and `--shm-size 8g`: the tensor-parallel workers' message queues and the KV connector's staging buffer live in shared memory. 8 GB is the value validated on this host.
- `--memory 52g`: on a 64 GB host the engine plus the 4 GiB CPU KV staging buffer must leave room for the OS and the clients. Uncapped, an engine swap has driven the host into swap-less thrash.
- `--oom-score-adj -800`: if the host runs out of memory, the kernel should kill a client, not the engine.

```bash
-e VLLM_ATTENTION_BACKEND=FLASHINFER
-e VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=268435456   # 256 MiB (128 MiB one card)
-e CUDA_MODULE_LOADING=LAZY -e PYTHONHASHSEED=0
-e TORCHINDUCTOR_COMPILE_THREADS=8 -e MAX_JOBS=4 -e FLASHINFER_NUM_COMPILE_JOBS=4
-e NCCL_P2P_LEVEL=SYS                                 # two cards only
-v $CACHE_DIR/torch_compile_<profile>:/root/.cache/vllm/torch_compile_cache
-v $CACHE_DIR/triton:/root/.triton/cache -v $CACHE_DIR/inductor:/root/.cache/inductor
-v $CACHE_DIR/flashinfer:/root/.cache/flashinfer
-v $L2MNT:/l2 -v $MODEL_DIR:/model -v $DRAFT:/draft:ro
```

- FlashInfer is the only backend with the sm120 NVFP4 KV path and the XQA decode kernel. The workspace cap bounds the lazily allocated autotune buffer; the two-card DFlash2 shape needs 256 MiB because the target and the drafter both allocate scale scratch.
- The compile-job caps and the persisted caches exist because an unbounded first-forward JIT once consumed all 64 GB of host RAM. One torch.compile cache per profile, because the captured graphs differ between KV dtypes.
- `PYTHONHASHSEED=0` makes the prefix-cache hashing and the tier namespace stable across boots.
- `NCCL_P2P_LEVEL=SYS` lets NCCL use the patched driver's peer-to-peer path. Decode allreduces go through vLLM's custom allreduce regardless, so this affects NCCL-carried collectives only.

## Model and parallelism

```bash
--model /model --served-model-name qwen3.8-27b qwen3.6-27b --trust-remote-code
--tensor-parallel-size 2                # two cards only
--kv-cache-dtype fp8_e4m3               # nvfp4 on one card
--gpu-memory-utilization 0.92           # 0.955 one card; 0.90 two-card MTP
--max-model-len 262144 --max-num-seqs 8 --max-num-batched-tokens 8192
```

- `fp8_e4m3` KV on two cards: DFlash2 drafts read the target's KV non-causally, and that read path is not yet clean on NVFP4 pages at TP=2 (the candidate in [`scripts/serve-nvfp4-candidate.sh`](../scripts/serve-nvfp4-candidate.sh) works around it by disabling XQA). fp8 KV costs +0.13 percentage points of perplexity versus bf16 KV and does not grow with context.
- Utilization is set per shape by the pool band the launcher asserts: 0.92 gives 654,491 or 628,798 tokens on this shape (the profiler's activation estimate is bimodal). Higher values OOM inside the FlashInfer autotuner on the first new batch shape, which no boot-time probe can see.
- `--max-num-seqs 8`: on this shape admission is bounded by the KV pool, not by this flag. 16 fits for short requests and raises aggregate decode about 34% at the cost of per-stream speed; 64 OOMs the speculative sampler at util 0.92. Measured in `results/2026-09-02-r159-conc-b`.
- `--max-num-batched-tokens 8192`: the prefill chunk. Larger chunks want more autotune workspace than the utilization leaves.

## Speculative decoding

```bash
--speculative-config '{"method":"dflash","model":"/draft","num_speculative_tokens":9}'   # two cards
--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":4}'              # one card
```

- Two cards: the [syvai W4A16 DFlash2 drafter](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16), 9 draft tokens. Tensor parallelism halves the verify cost, which is why 9 beats the 7 that was optimal on one card. The bf16 original drafter costs 6.5% code decode at 8 streams and 46K tokens of pool for no acceptance gain.
- One card: the checkpoint's own MTP head, 4 draft tokens. DFlash2 on one card costs two thirds of the context window because the drafter's KV is sized for the full context.
- Async scheduling stays on (the vLLM 0.28 default). `--no-async-scheduling` costs 21% prose and 29% code single-stream decode.

## KV disk tier

```bash
--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","cpu_bytes_to_use":4294967296,"offload_prompt_only":true,"secondary_tiers":[{"type":"fs","root_dir":"/l2","n_read_threads":16,"n_write_threads":4}]}}'
--enable-prefix-caching --mamba-cache-mode align
```

- vLLM's native offloading connector: a 4 GiB pinned CPU staging tier in front of a filesystem tier at `/l2`. It replaced LMCache in 2026-08 (no sidecar process, no 24 GiB pinned DRAM, no chunk-equals-block constraint).
- `offload_prompt_only`: offloading decode-phase blocks cost 1.8 tool-eval points in write stalls; prefix reuse only hits prompt blocks anyway.
- The filesystem tier has no eviction and no enforced capacity. The launcher requires the tier to be a fixed-size loopback filesystem ([`scripts/setup-native-l2.sh`](../scripts/setup-native-l2.sh)), refuses to boot with under 5 GB free, wipes stale namespaces, and stamps the tier with the checkpoint name because the tier's namespace is derived from the mount path rather than the weights. [`scripts/tier-evict.sh`](../scripts/tier-evict.sh) adds watermark eviction as a timer.
- `--mamba-cache-mode align` packs the GDN state into the unified KV block so the hybrid's state can be prefix-cached and offloaded. Worth about 3 tool-eval points with speculative decoding.

## Chat, tools, sampling

```bash
--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml
--override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20}'
--default-chat-template-kwargs '{"preserve_thinking":true,"reasoning_effort":"medium"}'
--limit-mm-per-prompt '{"image":16,"video":0}'
```

- `--limit-mm-per-prompt`: 16 images per request on the daily since 2026-09-03 (was 4). Raising the count costs nothing at boot (the profiler encodes one maximum-size image whatever the count; measured at 16 in `results/2026-09-03-r161-images`). A per-image pixel cap via `--mm-processor-kwargs max_pixels` shrinks the profiler reserve and OOMs the autotuner at util 0.92. Launcher knobs: `MMLIMIT`, `MMKW`.
- The Qwen3.8 template prefills `<think>` and injects a reasoning-effort line; `qwen3` parses the reasoning, `qwen3_xml` parses the tool calls (`hermes` drops them). The tool parser stays on even for clients that parse text themselves: a request that carries `tools` without `tool_choice` is rejected without it.
- T=0.6 over the model default T=1.0: 90.5 ± 2.1 versus 87.8 ± 1.3 on tool-eval (2026-08-14). A later A/B at the recommended T=1.0 settings confirmed the override (`results/2026-08-27-recsettings`).
- `preserve_thinking` keeps earlier `<think>` blocks across turns when the client resends them in the `reasoning` field. Effort `medium` is the default; `xhigh` livelocked an evaluation engine once and scored flat on SWE-Bench.

## Host

- GPU: persistence mode on, power limits 600 W and 575 W, memory clock offset +4500 MHz on both cards, core stock: [`scripts/gpu-tune.sh`](../scripts/gpu-tune.sh). The legacy NVML offset call silently no-ops on driver 610; the script uses the clock-offsets API and verifies by readback.
- Driver 610.57.04 with the QuixiAI peer-to-peer modules: [`scripts/gpu-p2p-610.sh`](../scripts/gpu-p2p-610.sh).
- Swap off. A host that runs out of memory next to pinned buffers hangs instead of failing.
- [`scripts/setup-earlyoom.sh`](../scripts/setup-earlyoom.sh): kills the largest client process at 5% available memory, never the engine, Docker or k3s.
- Kernel panic on hung task or soft lockup, so a wedged host reboots instead of staying wedged.
