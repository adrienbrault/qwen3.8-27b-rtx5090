# History: how the served configuration got here

## Daily lineage: what each daily was, and why the next replaced it

Newest first; the top row is the served configuration. Numbers for every switch are in the sections below and in [../bench/RESULTS.md](../bench/RESULTS.md).

| daily | weights · KV | pool | why it took over |
|---|---|---|---|
| **+ 7 draft tokens instead of 9 (2026-09-05 14:xx UTC, the served configuration)** | *(same weights, KV)* | **1,052,277** | The speculative-length ladder (R197, `results/2026-09-05-r197-spec-ladder`) over 6 to 11 DFlash2 draft tokens and the MTP head at 3 and 4: at 8 and 16 streams 6 and 7 beat 9 by 10 to 23% tokens per second, at 1 stream 9 keeps 11% on code. The 2026-09-04 retraction of 7 rested on a per-boot draw that R193d later attributed to the Triton autotune. The block drops to 1,552 tokens and the disk tier is wiped once (every tier hash carries the block size). |
| **+ FlashInfer's `pcie_ipc` all-reduce as the two-card decode all-reduce (2026-09-05, until 14:xx UTC)** | *(same weights, KV, pool)* | **1,020,596** | One 359 KB image layer (patch 0138) and one env knob, asserted at boot. Knob on against knob off on the same image, the same night ([R185](../bench/RESULTS.md#r185-flashinfers-pcie_ipc-all-reduce-vendored-into-the-served-image-as-an-opt-in-tp2-all-reduce-patch-0138-46-code-and-54-prose-single-stream-decode-31-at-30k-context-26-at-16-streams-the-8-stream-tokenss-flat-at-49-stepss-numerics-identical-to-the-kernel-off-control-2026-09-04-results2026-09-05-r185-pcieipc-scriptsr185-pcieipcsh-scriptsr185b-pcieoff-fidsh)): code c1 +4.6%, prose c1 +5.4%, prose at 30K +3.1%, 16 streams +2.6%, 8 streams flat in tokens/s and +4.9% in step rate; numerics identical on every paired ruler. The all-reduce was the decode-step cost that grew with concurrency (R183, R184). Rollback is the previous launcher without the layer. |
| **vLLM 0.29 nvfp4-KV route on the two-card DFlash2 shape (2026-09-04 → 09-05, the base of the served configuration)** | RedHat W4A4 NVFP4 · nvfp4 KV (block 1,584 with the bf16 GDN state since 16:24 UTC, 2,944 with fp32) | **1,020,596** at 16 sequences with the bf16 state (903,793 with fp32; 937,795 at 8 sequences), pinned 13.98 GB/card + 300 GB LRU-capped disk tier | The 0.29 program (RESULTS R164–R174): nvfp4 KV on `sm_120` rebased onto v0.29.0rc2, the pool pinned in bytes because the util path sizes it before graph capture, drafter CUDA graphs, embedding table offloaded to pinned host RAM, prefix-cache reuse under DFlash restored, LRU eviction added inside vLLM's fs tier, and FlashInfer pinned to 0.6.16.post3 because its split-KV prefill path is the closest of three to bf16. Paired against the fp8 shape the same day: +43% pool, disk tier serves 131K/220K prompts (4/4 vs 0/4), about −5% decode at 8 streams and at 30K context, dense top-1 vs bf16 92.8% vs 93.1%. A shorter draft (ns7, +12.5% c8) was retracted on a bf16 decode-path ruler. |
| **RedHatAI NVFP4 checkpoint on the two-card DFlash2 shape (2026-09-02 → 09-04, now the rollback)** | RedHat W4A4 NVFP4 · fp8 | **654,491** @0.92 (628,798 on the other profiler mode) + disk tier | Only the checkpoint changed. A bf16-anchored fidelity ladder over nine NVFP4 checkpoints put gittensor at +4.46% perplexity from bf16, last of the nine, and RedHat at +0.38%; on bf16-generated agentic turns gittensor diverged 2.5× more often (flips 8.6% vs 3.4%). Cost on the same shape: decode −6% c1 / −7% c8, prefill −14%, pool −12%. Gates: tool-eval 90.8 ± 0.5, needles 9/9, GSM8K 0.8583. `results/2026-09-01-r156-bf16-ladder`, `results/2026-09-02-r156-promote-redhat`, [R156-DECISION.md](R156-DECISION.md). |
| **DFlash2 + fp8 KV across two RTX 5090s, TP=2 (2026-08-31)** | gittensor W4A4 NVFP4 · fp8 | **711,281** @0.90 at promotion; 746,849 on the tuned form + disk tier | A second card was added (driver 610.57.04 with peer-to-peer modules). DFlash2 with the syvai W4A16 drafter at TP=2 measured code c1 298.9 t/s (+71% over one card), decode flat from the surface to 100K, prefill +49% at 100K; tool-eval 90.2, GSM8K 0.8417, needles clean. Draft length 7 → 9 the same day (code c1 325). Known cost: prefill −15% at 8K to 30K prompts. `results/2026-08-31-r132-vllm-dflash-tp2`, `r133-dflash-quality`, `r142-matrix`. |
| **vLLM v0.28.0 + patches-v0280: nvfp4 KV + XQA decode + MTP ns4 + native disk tier (2026-08-28)** | gittensor W4A4 NVFP4 · nvfp4 | **381,300** @0.955 + disk tier | Off the nightly onto the v0.28.0 release. vLLM's native OffloadingConnector replaced LMCache (no sidecar, no 24 GiB pinned DRAM, no chunk-equals-block constraint) on a fixed-size 200 GB loopback tier; the NVFP4 KV port was rebased with the XQA decode kernel, which removed the nvfp4 decode penalty; async scheduling on (+21% prose, +29% code c1). Tool-eval 90.0 ± 1.4 after `offload_prompt_only` removed the tier's 1.8-point write-stall cost. `results/2026-08-28-r108-promote`, `results/2026-08-29-r113-tuning`. |
| **gittensor NVFP4-RTX5090 checkpoint (2026-08-22)** | gittensor W4A4 NVFP4 (GDN projections quantized) · nvfp4 | **397,982** @0.93 + LMCache tiers | Checkpoint A/B on the unchanged engine: the saka checkpoint kept the 48 GDN layers' projections in bf16 (about 11 GB read per decode step); gittensor quantizes them. Pool 312,189 → 397,982, decode c1 143–150 → 178 t/s, c4 339–363 → 405, needles 4/4, prefill equal. `results/2026-08-22-sweep-ab`. Provenance: the parent HF repo `-RTX5090` later swapped its `lm_head` to BF16 and republished the original NVFP4-head recipe as [`-LMHead4`](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090-LMHead4), verified bit-identical to the checkpoint served here; every 2026-08 number in this repo was measured on it. |
| **Qwen3.8-27B tiers + NVFP4 KV on the V2 model runner (2026-08-21 evening)** | W4A4 NVFP4 · `nvfp4` KV | **309,090** @0.93 + tiers | The NVFP4 KV cache that R77 measured on the plain profile (+37% pool, overlay numerically required) composed with the LMCache tiers without new page code. The rc4 group edits work by byte accounting, so the only requirement was chunk = the nvfp4 unified block (2864; the launcher hardcoded fp8's 1616 until that was corrected). Gates: needles 10/10 + 10/10 to 100K with real retrieves, restart-proof 40K 6.3 → 2.5 s / 60K 11.6 → 3.3 s, "sean gate" (the concurrent-loader needle gate) 15/15 + 15/15, decode c1 141 / c4 339 / c8 353 (fp8 tier: 152 / 360 / 367), prefill 12.8K at 8K (mnbt 5727), tool-eval (tool-calling benchmark, tool-eval-bench) 92 ± 1.4. Image `tiers-nvfp4kv` = `Dockerfile.nvfp4kv` built FROM the tier image. Post-promotion 69×4, vision, SO probe and a soak are pending. R80/R81. |
| **Qwen3.8-27B tier-rc4 on the V2 model runner + nightly `ba07e4a48` (2026-08-21)** | *(same tiers, same patches)* | **209,859** @0.95 + tiers | The DFlash2 audition ([DFLASH2.md](archive/DFLASH2.md)) showed the speed came from the runner and the newer nightly, not the drafter. The same stack was re-platformed: `patches/rc4/Dockerfile.rc4` on the 08-21 nightly digest, `VLLM_USE_V2_MODEL_RUNNER=1`, fresh L2 namespace. Measured against the V1 daily the same hour: decode **c1 152 / c4 360 / deep-30K 143 t/s (+19% / +16% / +20%)**, pool +4%, needles 10/10+10/10 to 100K with L2 warm revisits 0.5–1.5 s, restart-proof L2 3.3/4.7 s, "killer" (the concurrent burst gate) 8/8, tool-eval 91 ± 0 (vs the 92.5 ± 0.7 best). Nightly defect: a flaky spawn-child `ImportError` (undefined cutlass symbol) on 1 in 6 boots, and a retry boots clean. Keep util 0.95 (V2 at 0.98 OOMs in the fp4_gemm autotuner). [V2RUNNER.md](archive/V2RUNNER.md). |
| **Qwen3.8 on vLLM nightly 0.27, plain (2026-08-15)** | *(same weights)* | **207K** @0.98 | The frozen 0.23 base was dropped. No patches survive: #42603/#44993/parser-workaround/async-crash are all obsolete upstream. Still true: `--mamba-cache-mode align` is worth ~3pts of 69×2 (91 ± 0 with, 87–88.5 without, from hybrid GDN state × spec decode) and async-off ~1pt. Perf: c1 123 / c8 322 tok/s, prefill 13.0K @8K. vLLM-native DSpark beats MTP +12–40%, but its draft KV caps context at ~64K, so it stays an opt-in profile. **The tier profile went temporarily back on the 0.23 image** (same day: agentic workloads re-prefill everything without LMCache, and the native prefix cache measured 0.0% hit under 4×50K contexts) until LMCache 0.5.3 is rebuilt against nightly. |
| **Qwen3.8-27B saka MTP-NVFP4 (2026-08-14)** | *(same engine + tiers)* | **214K** @0.95 + tiers | Qwen3.8 was released 2026-08-14 shape-identical to 3.6 (`qwen3_5`). The saka checkpoint reproduces natfii's W4A4 recipe byte-for-byte in size, so the whole profile carried over: same image, same patches, same pool. Promotion checks ("gauntlet"): plain 69×2 pairs **91/90** (3.6 plain ~89.8), tier 69×4 **87.8 ± 1.3** at T=1.0 (3.6: 89.0 ± 1.4 at T=0.6; CIs overlap), cold near-full burst 8/8, 60K needle, vision, parallel tool calls. Deltas: `deepseek_r1` reasoning parser (prefilled `<think>` plus a reasoning-effort system line), fresh L2 namespace, alias `qwen3.6-27b`. The T=0.6 override was kept after a same-protocol control: tier 69×4 **90.5 ± 2.1** at T=0.6 (equal or better vs 3.6's 89.0 ± 1.4, point estimate +1.5) vs 87.8 ± 1.3 at the model-default T=1.0. The unsloth fp8-attention NVFP4 was rejected (+2 GB weights, a 132K max-len ceiling). SGLang DSpark measured demo-only on 32 GB (~140 tok/s c1 / 197–293 aggregate c2 in a 6.7K-KV no-vision shape; the "206 tok/s" figure needs an unpublished SPS table). |
| + LMCache DRAM/NVMe tiers (2026-07-20) | *(same engine)* | **214K** @0.95 **+ ~2.4M tiered** | Six local patches made tiered KV faithful on this fp8 hybrid (cross-restart needle plus 69×2 = **89** vs a ~89.8 baseline). It trades 25K hot tokens and `mnbt` 4096 for ~2.4M tokens of second-chance capacity and a warm start after restarts: a 60K revisit costs 2 s (DRAM) or 4–7 s (NVMe) instead of an 11–13 s re-prefill. Validated by an 858-cycle soak (flat VRAM, L2 stable under its cap). [`../scripts/legacy/serve-plain.sh`](../scripts/legacy/serve-plain.sh) keeps the row below available. |
| natfii NVFP4 W4A4 · fp8_e4m3 + FlashInfer + MTP `ns=4` (2026-07-19) | NVFP4 W4A4 · fp8 | ~239K @0.98 | **Prefill 3.4×** (13.5K vs 4.0K t/s @8K, native Blackwell FP4 GEMM vs Marlin dequant), deep-concurrent sustained 2.2× (148 vs 67 t/s at pp30K×c8 tg512), cold 60K context 10 s vs 23 s, at equal 69×2 quality (~90, 4 trials each side; the W4A4 activation cost was bounded at ≈1 pt by a chimera A/B, and natfii's calibration covers it). It passed the full promotion checks including a 106-cycle soak and a 0.98 combined-wave battery. Pool is 11% smaller than AR's 270K (heavier MTP head plus FP4 scales), traded for re-prefilling 3× faster. |
| Lorbus INT4-AutoRound · fp8_e4m3 + FlashInfer + MTP `ns=4` (2026-07-18) | INT4-AutoRound · fp8 | ~270K @0.96 | Flat deep decode: fp8+FlashInfer has no decode crater at depth, where the custom TurboQuant kernel drops. Largest pool measured to that point; **MTP `ns=4`** restored by [PR #42603](https://github.com/vllm-project/vllm/pull/42603); tool-eval 90. It dropped the experimental TurboQuant KV kernel for the stock fp8 path. (A one-day 0.98/287K promotion was reverted the same night: serve-time autotune OOM, [GOTCHAS.md](GOTCHAS.md) #8.) |
| turboquant_4bit_nc (NVFP4) + MTP `ns=3` (2026-07-15) | NVFP4 · TQ 4-bit K/V | ~235K | +42% pool over k8v4, once the "4bit_nc destroys retrieval" **0/8** was traced to the async×spec KV confound and fixed with `--no-async-scheduling`. Decode still drops sharply at deep single-stream, the custom-kernel cost. |
| turboquant_k8v4 (NVFP4) | NVFP4 · TQ 8-bit K/4-bit V | ~165K | +21% pool over fp8 at fp8-equal retrieval quality (8-bit keys). |
| fp8_e4m3 (stock nightly) | NVFP4 · fp8 | ~136K | The original baseline: flat deep decode, no patches, smallest pool. |

Summary of the sequence: the fp8 baseline's flat deep decode survived every generation; PR #42603 added working `ns=4`; AutoRound added quality and pool; NVFP4 W4A4 used the GPU's native FP4 compute, and was the first daily where prefill rather than decode or capacity got a generational jump; and the KV tiers moved capacity out of the 32 GB box entirely, the one axis a bigger GPU would otherwise have been the only answer to.

## 2026-09-05: seven draft tokens

The draft length was last read on 2026-09-04 (R173/R173b/R173c): 7 gave +12.5% at 8 streams and was retracted because it sat twice as far from the bf16 decode reference at 30K. R193d then showed a difference of that size between two boots of one configuration, the runtime Triton autotune, so the retraction rested on a reading the protocol voids. The 2026-09-05 ladder (R197, `results/2026-09-05-r197-spec-ladder`, one boot per length on the served route with the served flags) read 6, 7, 8, 9, 9, 10 and 11 DFlash2 draft tokens and the MTP head at 3 and 4. Promoted at 14:xx UTC: 7 draft tokens, block 1,552, pool 1,052,277, the launcher wipes the disk tier once because every tier hash carries the block size. The single-stream code rate drops from 306 to 273 t/s; 8 and 16 streams gain 10 to 23%.

## 2026-09-05: batch-sharded sampling

vLLM's `--enable-batch-sharded-sampling` has each tensor-parallel rank sample its half of the batch from local logits and exchange the tokens, replacing the per-step logits all-gather (19.9 MB at 16 streams). R187 saw it as the one flag outside the step-rate band; R191 read its numerics on different compile artifacts and was withdrawn once R193 showed the artifact lottery; patch 0147 removes the flag from the configuration hash so both settings share one artifact, and R193b, R193c and R193e then read the sharded sampler bitwise against the unsharded one at temperature 0 (R193e under the full protocol: one artifact, autotune pinned on both engines). Steps per second +2.6% at 8 streams, +4.5% at 16, single-stream unchanged.

Promoted at 11:03 UTC (R195, `results/2026-09-05-r195-promote-bss`): boot asserts the sampler's log line and the patch marker; pool, needles, capacity and tool-eval (90 ± 0.8) as before; 8 and 16 streams at 378 and 504 steps/s. One behavioural change for clients: a request that passes a seed at temperature above 0 gets a different sample stream than the unsharded sampler gave. Rollback `scripts/serve-r189-daily.sh`.

## 2026-09-05: the `pcie_ipc` all-reduce layer

R183 put the two-card decode tax on vLLM's custom all-reduce (15% of the step at 1 stream, 33% at 16) and R184 measured FlashInfer main's `pcie_ipc` kernel 24% to 36% faster at the served row counts. Patch 0138 vendors that kernel into the served image as an opt-in TP=2 all-reduce for bf16 inputs up to 320 rows, with the R184 launch configurations fixed and no boot autotuning. R185 and R185b ran it on the image with the knob on and off the same night (`results/2026-09-05-r185-pcieipc`): +4.6% code and +5.4% prose single-stream decode, +3.1% at 30K context, +2.6% at 16 streams, the 8-stream tokens/s flat at +4.9% step rate, and identical numerics on the ctx0 and 30K decode rulers and the agentic ruler. Promoted 2026-09-05: [../scripts/serve-r168-daily.sh](../scripts/serve-r168-daily.sh) boots the `...-fi0616-pcieipc` image with `VLLM_SM12X_PCIE_IPC_AR=1` and fails the boot without the enabled line and the backend order. The live-port gates (pool, capacity probes, needle gate through the tiers, decode, tool-eval) run in [../scripts/r189-promote-pcieipc.sh](../scripts/r189-promote-pcieipc.sh) after the queued experiments; its numbers land in [../bench/RESULTS.md](../bench/RESULTS.md) as R189.

The promotion was gated on the serving port at 09:01 UTC (R189, `results/2026-09-05-r189-promote-pcieipc`): first-try boot at the pin, pool unchanged, 4 of 4 needles cold and 4 of 4 re-served from the tiers written before the restart, five 100K prompts co-resident, tool-eval 91.2 ± 1.3, zero engine errors. The launcher without the layer is the rollback.

## 2026-09-04: the vLLM 0.29 nvfp4-KV route, and the bf16 GDN state

What the route trades against the two-card fp8 shape it replaced, measured the same day on the same box:

- The pool grows from 654,491 (or 628,798, the fp8 shape's profiler was bimodal) to 903,793 at 16 sequences, and the pin makes it the same on every boot. The daily serves 16 sequences since 2026-09-04 (8 before), because 16 streams gave +34% aggregate decode over 8 on the fp8 shape (`results/2026-09-02-r159-conc-b`); the pin for 16 costs 34K tokens of pool.
- The disk tier serves. On the fp8 shape every revisit of a 131K prompt was recomputed: a tier hit is promoted through the CPU tier, 331 fp8 blocks did not fit its 4 GiB, and with 16 GiB the v0.28 lookup still served 0 of 4 needles. On the served route, needles at 131K and 220K were served 4 of 4 after a 12-prompt flood and 4 of 4 after a fresh boot, in 1.4 to 2.9 s against 25 to 57 s cold (`results/2026-09-04-r172-cputier`).
- The fidelity cost is the NVFP4 KV cost measured in R156 and unchanged across vLLM versions: 0.3 points of top-1 agreement and 0.4 points of perplexity against bf16 ([FIDELITY.md](FIDELITY.md)).
- Decode, paired against the rc2 image with fp8 KV (`results/2026-09-03-r169-rc2`, decode_ss): code 1 stream 245 vs 227, 30K context 155 vs 164, code 8 streams 1,110 vs 1,153 (`results/2026-09-04-r173-c1-opt`). Prefill at 2K: 8,757 vs 8,548 (`results/2026-09-03-r166-gates`). Tool-eval on the same instrument: 91 vs 88.

The route scored SWE-Bench Verified 388/500 = 77.6% on 2026-09-04 (mini-SWE-agent 2.4.6, official harness, one attempt, 12 workers, 3 h 37 min, `results/2026-09-02-miniswe-rh-r174-nvfp4`). The fp8 shape scored 386/500 = 77.2% on 2026-09-03 with the same harness (`results/2026-09-02-miniswe-rh`). Paired on the 497 instances both runs completed, 351 were resolved by both, 34 only by the fp8 shape and 37 only by the served route, so the two score the same. Earlier one-card configurations scored 331/500 = 66.2% (2026-08-21, saka checkpoint, R2E-Gym scaffold) and Terminal-Bench 2.1 50/89 = 56.2% (2026-08-23, gittensor checkpoint, Harbor with terminus-2).

Since 16:24 UTC the same day the linear-attention state is cached in bf16 (R182): the attention block drops from 2,944 to 1,584 tokens, the pool reads 1,020,596 at the same pin, and a request costs half as much of it (3.4% instead of 6.4% for a short request). The measurement and the fidelity cost are in [DESIGN.md](DESIGN.md#what-a-request-costs-in-the-pool) and [FIDELITY.md](FIDELITY.md); the fp32-state launcher `serve-r174-daily.sh` is the rollback.

## 2026-07-20: LMCache DRAM/NVMe KV tiers added to the daily

Same natfii engine as the section below, plus the tiered KV offload: util 0.95, pool 214,084, +24 GB pinned DRAM (~245K tok) +200 GB NVMe (~2.13M tok, restart-proof), at quality parity (69×2 = 89 vs ~89.8; later confirmed **89.0 ± 1.4** over a 69×4 re-run, [cross-trial stats](../bench/RESULTS.md#tool-eval-cross-trial-statistics-694-on-the-tier-daily-2026-07-22)) after six local patches. That campaign took four rounds of tier profiles that validated but were silently wrong; it is written up in [LMCACHE.md](archive/LMCACHE.md) and [../patches/lmcache/README.md](../patches/lmcache/README.md). The section below documents the natfii engine promotion this profile is built on; its util-0.98/239,436 numbers are now the plain (no-tiers) profile.

Same era, the last TurboQuant audition. With boot unblocked on the modern stack, `turboquant_4bit_nc` was auditioned once more against the promotion gates: it reached a **563,888-token pool** (2.6× the daily's) and failed both disqualifying gates, returning a wrong 60K needle (the hybrid buffer co-location corruption class, unfixed) and dying under the `pp8192×c8` concurrency burst. A pool that size is not usable if retrieval is wrong, so TurboQuant KV is closed permanently for this hybrid: revival would need a state-corruption fix, a concurrency fix, a deep-decode-crater fix, and new LMCache serde kernels for its packed layout, against a working fp8 tier stack. The 2026-07-15 un-rejection below remains true as far as it went (the 0/8 was the async confound), but it was not the whole cause. [REJECTED.md](REJECTED.md) records the rejection.

## 2026-07-19: the daily moved to natfii NVFP4 W4A4, for prefill

The daily moved to the [natfii W4A4 NVFP4 export](https://huggingface.co/natfii/Qwen3.6-27B-VLM-NVFP4-MTP) at util 0.98, pool 239,436. The case: equal quality (69×2 pooled 89.8 vs 87.8, 4 trials each), **prefill 3.4×** (13.5K vs 4.0K t/s @8K, because W4A4 dispatches Blackwell's native FP4 GEMM where W4A16 runs bf16 GEMM plus Marlin dequant), deep-concurrent sustained 2.2× (148 vs 67 t/s at pp30K×c8 tg512), cold-60K TTFT 10 s vs 23 s. Cost: pool −11% (natfii's MTP head is 0.79 GiB vs AR's 0.28, plus FP4 scale tensors).

Three findings from the campaign are worth keeping:

- The W4A4 quality question was settled by construction. A chimera checkpoint was built: natfii's W4A4 MLPs plus NVIDIA's fp8 attention projections, merged tensor-by-tensor into one MIXED_PRECISION export, with both kernel classes co-dispatching in one graph. All three variants were scored on the full 69×2. Chimera 90.0, natfii ~89.8, NVIDIA 91.0: the entire activation-quant cost is ≈1 point, natfii's calibration covers it, and no attention/MLP remix beats it without requantizing. The chimera was archived the day it answered the question.
- The util ceiling is model-specific. 0.98 OOM'd the AR daily at serve time (addendum below) but passes on natfii: lighter margin pressure per shape, `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=134217728` capping the autotune workspace, and a boot-time `pp8192×c8` pre-warm so the allocation happens before traffic. Validated by the full battery at 0.98 (needle, both concurrent burst shapes, 8× text flood, 8× vision burst), then two simultaneous combined waves on a cold engine, then a 106-cycle overnight soak. Steady-state floor ~130–190 MiB, no drift.
- The checkpoint shipped a defect: `tokenizer.json` with `truncation: {max_length: 8192}` baked in, a calibration leftover. Text is fine; multimodal above 8K tokens returns hard 400s. `serve-tier-rc4.sh` (then `serve.sh`) nulls it at every launch, because a re-download reinstates it.

## Addendum, 2026-07-18 → 2026-07-19: util 0.98 lived one day

The 0.98/287K promotion below was reverted the same night it shipped. 0.98 boots clean and survives every boot-time burst probe (near-pool-full distinct-prompt text, 8× concurrent 4-image vision, mixed), and then OOM-dies in production the first time the fp4-GEMM/FlashInfer autotuner meets a genuinely new deep batch shape: a fresh `pp8192 × c8` wave allocates ~266 MiB of lazy benchmark workspace at serve time against ~600 MB of margin (2/2 reproducible, `EngineDead`, full log captured). No boot-margin method can see this failure, because the workspace only materializes when traffic produces the shape. The AR daily then ran **util 0.96, pool 270,422** (~1.2 GiB margin), validated against that exact burst shape plus the full burst battery, and `benchy --pp 8192 --concurrency 8` is a permanent member of the promotion checks. (The natfii daily above later re-qualified at 0.98, with the workspace-cap env, a boot pre-warm, and this exact battery; the ceiling is model-specific, the method is not.) `mnbt 8192` (a +15–20% concurrent-decode candidate) wants ~486 MiB of workspace and only fits at ≲0.94: measured, parked.

The daily config in this repo (patched TurboQuant image + `turboquant_4bit_nc` KV + `--no-async-scheduling`) did not arrive in a straight line; it took **two** reversals. Stock fp8 shipped for weeks first, with the TurboQuant image treated as experimental over what looked at the time like intermittent memory corruption. That turned out to be a noisy detector plus 4-bit-key quality loss, so the config moved to `turboquant_k8v4` (8-bit keys). The 4-bit-key quality loss was then found to be largely a third confound, vLLM's async scheduler corrupting KV under speculative decode, and one flag (`--no-async-scheduling`) makes the denser `turboquant_4bit_nc` clean, which is how it became the daily. This page records that sequence, newest chapter first, plus the genuine bug it uncovered: a discarded out-param under CUDA-graph capture, which the patch stack does fix.

## The daily moved to Lorbus INT4-AutoRound + fp8 + MTP ns=4 (the PR #42603 sync workaround)

2026-07-18: the daily moved off TurboQuant KV to a simpler, stock path, Lorbus INT4-AutoRound weights + `fp8_e4m3` KV + FlashInfer + MTP `ns=4`. The fp8 attention path is flat with depth (no decode crater), the pool is **~287K** at util 0.98 (larger than the 235K TurboQuant daily below), and tool-eval is 90. The blocker was a crash, and locating it took a long bisection worth recording, because almost every obvious fix was wrong.

The crash: MTP `ns≥2` + `fp8_e4m3` KV on Blackwell `sm_120` hits an illegal memory access at `rejection_sampler.py:267 parse_output` under any real concurrency (c≥4). Single-stream is clean, `ns=1` is clean, and `CUDA_LAUNCH_BLOCKING=1` masks it, which indicates a timing race.

The dead ends, each ruled out empirically with a fresh build and a concurrent reproduction:

- Device-wide `torch.accelerator.synchronize()` barriers in `gpu_model_runner`, placed after the accepted-state postprocess (before the proposer), before the `bookkeep` block, and inside `execute_model` right after the Mamba input staging. All three fired, as the logs show, and all three still crashed. So it is not an ordering race anywhere in the model runner's spec loop: a late sync cannot repair an already-wrong value.
- `--mamba-cache-mode all`, which skips the fused `align` Mamba postprocess kernel entirely, still crashed at the same pool. So it is not the fused Mamba postprocess.
- A draft-token sanitizer (drop any `id<0 or id≥vocab` before the rejection sampler, the approach in the closed upstream [#46574](https://github.com/vllm-project/vllm/pull/46574)): the drop warning never fired. The fault was never an out-of-vocab draft token.

The working hypothesis and the workaround, [vLLM PR #42603](https://github.com/vllm-project/vllm/pull/42603). An upstream search turned up [#40756](https://github.com/vllm-project/vllm/issues/40756) (same Qwen3.6-27B-FP8 model) and [#35288](https://github.com/vllm-project/vllm/issues/35288) ("MTP corrupted output at concurrency ≥ 4"), a known, still-open class: MTP × fp8 KV × Blackwell. The PR's hypothesis: the MTP draft loop in `vllm/v1/spec_decode/llm_base_proposer.py` writes shared cudagraph buffers (`input_ids`/`hidden_states`) then launches the draft forward reading them without a stream sync, so under concurrency the draft FlashInfer kernels read stale buffers. That locality, inside the proposer loop, is at least consistent with why every `gpu_model_runner` barrier missed it. The workaround is one `torch.accelerator.current_stream().synchronize()` after the writes. Upstream closed the PR unmerged: maintainers held that these operations should already be stream-ordered, that a forced sync may only perturb timing, and asked for a proven root cause. The claim here is therefore strictly empirical: on this profile the crash is 100%-reproducible without the sync and has never occurred with it, validated across concurrent c4/c8, deep pp90000×c4 and the full 69×2 tool-eval, and perf-neutral. Grafted as [`patches/install_pr42603_sync.py`](../patches/install_pr42603_sync.py); the upstream root cause remains unresolved.

Method, again: localize before fixing. `CUDA_LAUNCH_BLOCKING` named the class of fault (a timing race), and serializing one candidate edge at a time to prove where the fault was not is what pointed at the proposer's own loop. The upstream issue search turned a multi-day bisection into a one-line graft.

---

## turboquant_4bit_nc became the daily (the async×spec reversal)

2026-07-15: this reverses the "`turboquant_4bit_nc` destroys retrieval" call recorded in the next section. `turboquant_4bit_nc` (4-bit keys) had been rejected because it scored **0/8** on needle-in-haystack, and the conclusion drawn was "4-bit keys destroy long-context retrieval." That conclusion rested on a confound. The actual cause was vLLM's async scheduling × speculative decode interaction, a batch-row-mapping desync ([vllm#42655](https://github.com/vllm-project/vllm/issues/42655)).

Mechanism: with MTP, every scheduler step emits a multi-token verify batch, so even a single in-flight request occupies multiple batch rows. vLLM's async scheduler computes its request-ID to batch-row mapping one step ahead of execution; under those multi-row verify batches the mapping desyncs, and KV is written to the wrong slots. 4-bit keys are far more sensitive to that corruption than 8-bit keys, so it appeared as **0/8 for `4bit_nc`** but only ~10% intermittent degradation for `k8v4`. That is why k8v4 looked fine and `4bit_nc` looked broken while both were being corrupted by the same bug.

The fix is one flag, `--no-async-scheduling`. With it, and after confirming the engine loaded genuine `4bit_nc` (the log shows a ~235K pool, not a silent 165K `k8v4` fallback), `turboquant_4bit_nc` is completely clean:

| test (with `--no-async-scheduling`) | turboquant_4bit_nc |
|---|---|
| single-stream needle-in-haystack @9K / 20K / 40K | **8/8 / 8/8 / 8/8** |
| high-pressure concurrent (3 rounds × 30 needles, 6 background loaders) | **90/90** |
| tool-eval-bench v2.1.0 (matched protocol, hardmode) | **89** (parity with k8v4) |

That 90/90 is the same test that was said to kill all 4-bit-KV kernels. It passes.

Why it became the daily: `4bit_nc` buys **~235K pool → 200K usable context** (+42% pool, +25% context vs `k8v4`'s 165K→160K) for a small decode cost. Fresh same-session llama-benchy, both async-off: pp512 decode c1 137→133 (−3%), c8 467→435 (−7%); pp4096 c1 145→126 (−13%), c8 parity. The 4-bit-key dequant (Lloyd-Max codebook plus per-GQA-head norm-correction; the inverse Hadamard is hoisted to one per-query GEMM, not per key) costs more than k8v4's cheap FP8-cast keys, worst at deep single-stream. It trades pool for a modest decode loss, and interactive coding is the low-concurrency, deep-context regime where the extra 40K of context is worth ~10% decode.

Broader implication: the belief that all custom 4-bit-KV kernels corrupt under concurrency, which this repo's [`fix_spec_guard.py`](#the-discarded-out-param-bug) patch partly addressed, was very likely this async×spec scheduler bug rather than the kernels. Whether the existing guard patches are still needed with async scheduling off has not been retested, so they stay in the stack for now. The [`fix_spec_output.py`](#the-discarded-out-param-bug) out-param fix (#40914) is a genuinely separate bug and stands regardless. `--no-async-scheduling` is a required companion flag, not a replacement for the patches.

`turboquant_k8v4`, the prior daily documented in the next section, remains a good decode-optimal middle ground: slightly faster than `4bit_nc`, with a smaller pool. It is no longer rejected as the fix for 4bit_nc; both are shipping presets now. fp8 stays the deep-context, high-concurrency batch alternative.

## turboquant_k8v4, the prior daily

> Superseded 2026-07-15 by `turboquant_4bit_nc` (above). Kept as the record of the first reversal, from "TurboQuant corrupts" to k8v4, and because k8v4 is still a shipping alternative. The claim below that "`turboquant_4bit_nc` scored 0/8, 4-bit keys destroy retrieval" is the call the section above overturns: that 0/8 was async×spec corruption, not the keys.

2026-07-15: this reverses the earlier call. For weeks stock fp8 KV was the daily and the patched TurboQuant image was treated as experimental, because it produced constant `!!!!` at 0% MTP acceptance under real agent sessions, diagnosed then as intermittent memory corruption. That diagnosis was wrong; there was never a corruption bug. A ~30-round investigation traced the `!!!!` to two mundane causes: (1) a soak-test degeneracy detector that false-positived on coherent replies to random-gibberish prompts, and (2) genuine long-context quality loss from quantizing the keys to 4 bits (`turboquant_4bit_nc`). Attention indexes on the keys, and 4-bit keys destroy long-context retrieval.

The fix was one flag: use `turboquant_k8v4` (8-bit keys / 4-bit values) instead of `turboquant_4bit_nc`. 8-bit keys preserve retrieval, and 4-bit values keep most of the density gain. Measured by fair needle-in-haystack retrieval: plant 5-digit codes in coherent filler and require an exact match:

| KV cache | 9K | 20K | 40K |
|---|---|---|---|
| `turboquant_4bit_nc` (4-bit keys) — *async on* | **0/8 across depths** | | |
| fp8_e4m3 | 6/6 | 8/8 | — |
| **turboquant_k8v4** | **8/8** | **8/8** | **6/6** |

The `4bit_nc` 0/8 above was measured with async scheduling on: the cause was the [async×spec desync](#turboquant_4bit_nc-became-the-daily-the-asyncspec-reversal), not the 4-bit keys. With `--no-async-scheduling`, `4bit_nc` scores 8/8 at all three depths. The k8v4 and fp8 rows still hold.

So the TurboQuant + MTP stack and its four patches are still real and still needed: the [discarded out-param bug behind the `!!!!`](#the-discarded-out-param-bug) was genuine and its fix stands. What is retired is the conclusion that the image was demoted to experimental because it corrupts. The daily image is the same clean TQ image, just `VLLM_TQ_PRESET=turboquant_k8v4` + `--kv-cache-dtype turboquant_k8v4`. It is faster single-stream, **+21% pool** (165K vs 136K tokens, in less KV memory), and matches fp8's retrieval and MTP acceptance. fp8 remains the alternative for deep-context high-concurrency batch serving (see [Benchmarks](../README.md#what-you-get)). At the time `turboquant_4bit_nc` was also recorded as a rejected variant on its 0/8 retrieval; the [section above](#turboquant_4bit_nc-became-the-daily-the-asyncspec-reversal) overturns exactly that, once the async×spec bug is removed.

## The discarded out-param bug

vLLM PR #40914 fixes TurboQuant's spec-decode routing. Its new branch ends:

```python
attn_out = triton_turboquant_decode_attention(...)
return attn_out          # ← the bug
```

But `TurboQuantImpl.forward()` is invoked as a mutated-out-param custom op (`unified_attention_with_output`). Under `FULL_AND_PIECEWISE` CUDA-graph capture the return value is discarded: the caller reads the `output` buffer. So the attention output stays stale or zeroed and the model decodes a constant token:

```
prompt: "def fib(n):"
output: "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
```

Every other branch of `forward()` writes the buffer; that one did not. The fix ([`patches/fix_spec_output.py`](../patches/fix_spec_output.py)):

```python
if output.ndim == 3:
    output[:N] = attn_out.to(output.dtype)
else:
    output[:N] = attn_out.reshape(N, -1).to(output.dtype)
return output
```

This is very likely why the PR passed on the author's Ampere box, where the eager/piecewise path does consume the return value, and fails on Blackwell, where full CUDA-graph capture is the default.

---

## Archive: the 2026-07 LMCache investigation (vLLM 0.24 / LMCache 0.5.1 / the 0.23 base)

Accurate for its time; superseded by the current profile wherever the two conflict (vision, util, chunk size, `ns`, image, KV dtype). Moved here from LMCACHE.md on 2026-08-21.

## Architecture

vLLM keeps its KV cache on the GPU and evicts on pool pressure; a revisit re-prefills from scratch (**5.8s** on a 40K-token session). [LMCache](https://github.com/LMCache/LMCache) adds tiers below the GPU:

```
on-GPU prefix cache   ≈ instant     (vLLM's own, evicts under pressure)
  └─ L1: 24 GB pinned host RAM   ~0.5s hit
       └─ L2: 150 GB SSD         ~2.7s hit   (~2M tokens; survives restarts)
            └─ cold re-prefill    5.8s
```

Qwen3.6 is a hybrid (GDN/linear-attention + full-attention), so its mamba state must be stored as an opaque page. Only LMCache's `LMCacheMPConnector` does that ([LMCache PR #3613](https://github.com/LMCache/LMCache/pull/3613), in 0.5.1), and it needs an out-of-process `lmcache server` sidecar (ZMQ :5555) that owns the L1/L2 tiers.

## The image

```bash
IMAGE=lmcache-vllm:fixed
```

`lmcache/vllm-openai:latest-cu129` (vLLM 0.24 + lmcache 0.5.1) with one fix: it is CUDA-13-linked but ships no `libcudart.so.13`, so the `csrc` load fails. A multi-stage `COPY` takes `libcudart.so.13*` from `nvidia/cuda:13.0.1-runtime-ubuntu24.04`, then runs `ldconfig`. (`pip install nvidia-cuda-runtime-cu13` does not work: PEP 668, then no wheel.)

Do not use the nightly pairing (`latest-nightly-cu129`, lmcache 0.5.2-dev releases through at least 2026-07-06): it rejects this model's KV layout with `Unsupported EngineKVFormat: 10` on every store. The sidecar logs the error while serving continues, so the profile silently degrades to vLLM's in-GPU prefix cache only, and every tiered revisit measured that way is unreal. The first fix attempt, a hand-written format-10 transfer kernel ([patch, now withdrawn](../patches/README.md#lmcache-format-10-kernel-patch-separate-project)), made stores run but restored corrupted context: the needle vanishes after reload, tool-eval 88 → 47. The root cause, established later and not the first guess: vLLM's fp8 attention backend registers each hybrid-aligned page as ~100 contiguous 16-token kernel pages in a fused rank-4 tensor, while LMCache's kernel-page to logical-page regrouping only matched the rank-5 split-K/V layout, so it read the 16-vs-1616 slots/tokens ratio as compression and transferred one 16-token page per logical block, wrongly addressed and with zero errors logged. The GDN state pages were stored and restored fine all along, which is why output stayed fluent while distant facts vanished. LMCache `main` (≥ `0.5.2.dev66`, PR #4128) has the native format kernels, but as of `e38ee415` the regrouping gap is still there for the fp8 fused layout: bf16 hybrid passes a cross-restart needle because its raw page is already scheduler-sized, and fp8 hybrid does not. A stride-aware fix was in progress and has since landed as patches 0001/0002 in the current image. Whatever the setup: needle-test across a restart before trusting any external KV tier on a hybrid model, because hit counters and coherent output do not prove fidelity.

## Container flags

```bash
--entrypoint bash        # the image entrypoint is `vllm serve` — it would swallow the `bash -c`
--ipc=host               # CUDA-IPC across processes; without it, hangs at "Creating transfer context"
--memory 52g             # cgroup cap. The 24 GB L1 is PINNED host RAM — drop_caches before launch
-e MAX_JOBS=4 -e FLASHINFER_NUM_COMPILE_JOBS=4      # cap the sm120 fp4-GEMM JIT (see gotcha 1)
-v .../cache/flashinfer:/root/.cache/flashinfer     # MANDATORY persistent cache for that JIT build
```

Never add `-e PYTORCH_ALLOC_CONF=expandable_segments:True`. cuMem/VMM memory is not CUDA-IPC-exportable ([pytorch #165685](https://github.com/pytorch/pytorch/issues/165685), [vllm #29544](https://github.com/vllm-project/vllm/issues/29544)): the sidecar crashes importing the KV handles and never acks, and vLLM's `register_kv_caches` silently times out at 300s. This is the single most expensive gotcha here, because it read as version skew for days.

## The `lmcache server` sidecar

```bash
lmcache server --host 0.0.0.0 --port 5555 --chunk-size 1600 \
  --l1-size-gb 24 --l1-init-size-gb 2 --eviction-policy LRU \
  --worker-reap-timeout-seconds 0 \
  --l2-adapter '{"type":"fs_native","base_path":"/l2","max_capacity_gb":150,"num_workers":4}'
```

- `--chunk-size 1600` must equal vLLM's unified block size: **1600** with MTP `ns=3`, 1568 without (found by measurement, not documented; it is not 16). A mismatch gives "chunk size must be a multiple of vLLM block size".
- `--l1-size-gb 24` is the pinned-RAM tier. **It must exceed the hot working set / 0.8**, or LMCache's lookup breaks at the first missing chunk: LRU evicts the oldest session's head chunks first, that head-miss forces a full re-store, which evicts the next head, giving permanent thrash and 0% hits. Partial caching does not degrade gracefully; undersize this and the cache is inert, not merely smaller.
- `--worker-reap-timeout-seconds 0` disables the worker-registration reaper. The default 120s plus a lazily-started heartbeat means a long idle or blocked span gets reaped, after which the cache cannot recover: a permanent zombie with `found_count=0` and stores silently dropped at the worker's health gate.
- `--l2-adapter fs_native … 150 GB` is the SSD tier on a host directory mounted at `/l2`. ~2M tokens, and it survives container restarts. A 10×40K working set (29 GB) spills here with zero thrash.

## vLLM flags

```bash
--kv-cache-dtype fp8_e4m3
```
fp8 KV. (4-bit TurboQuant KV is not composed here: this profile prioritises retention and concurrency, where fp8 already wins.)

```bash
--kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both"}'
```
Routes KV through the MP connector (store and load). It is the only connector that handles this hybrid's mamba state. As a safeguard, vLLM 0.24 also accepts `kv_load_failure_policy: "recompute"` here; the default `"fail"` turns any load failure into a request 500.

```bash
--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'
```
MTP, composed with the cache. Upstream trackers call this composition unsupported, but it works here: the "MTP+LMCache crashes" reports were a VRAM burst-OOM (gotcha below), not the scheduler wall. Keep the default cudagraph mode: forcing `FULL_DECODE_ONLY` fails to cover MTP's verify-step shapes on 0.24 and collapses decode to **c1 46 / c8 179** (vs 118/450).

```bash
--gpu-memory-utilization 0.93
```
Not 0.94. The `lmcache server` process holds ~1.4 GB of VRAM (CUDA context plus IPC mappings) that `--gpu-memory-utilization` does not account for. At 0.94, a burst of concurrent cold prefills OOMs (74 MB free at crash) and kills the engine. That is the whole of the "MTP crashes with LMCache" report.

```bash
--max-model-len 120000
--max-num-batched-tokens 3199        # = 2*chunk - 1
```
`120000` is the opencode client's context. `--max-num-batched-tokens` must be `2·chunk−1` (3199 with chunk 1600; 3135 with 1568), because LMCache's MP connector requires batched-tokens ∈ [chunk, 2·chunk). This batched-token ceiling is why prefill runs ≈−10% vs the daily's 8192 path at depth.

```bash
--limit-mm-per-prompt '{"image":0,"video":0}'
```
Vision off. This is the one capability gap vs the daily. A vision-on variant is untested; `image:0` also buys a thriftier KV pool (163K tokens no-MTP, 124K composed).

```bash
--max-num-seqs 8 --mamba-cache-mode align --enable-prefix-caching --enable-chunked-prefill
--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml
--override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20}'
```
Identical to the daily; see [CONFIG.md](CONFIG.md) for each. `--mamba-cache-mode align` is what lets prefix caching work on the Mamba layers, and `qwen3_xml` is the correct tool parser (`hermes` drops the calls).

## Known issues

- Two stochastic failures shared by every composed run: an instant EOS at depth (`comp=1` empty, the documented temp-0.6 quirk) and a ~1-in-3 failure at pool capacity. This is noisier than the clean no-MTP LMCache run; a failure-rate probe is pending.
- The `engine_driven` transfer mode serves but scores 0 hits on this hybrid; use the default `lmcache_driven`.
- The no-MTP LMCache config (chunk 1568, no `--speculative-config`, util 0.90+) is the conservative fallback if MTP composition ever regresses: c8 517 (above the daily's 488), c1 69 (the 0.24 image cost plus no MTP).

## LMCache + k8v4: composes, but the persisted tier is lossy, not shipped

`turboquant_k8v4` KV (a prior daily; the later TurboQuant daily's `turboquant_4bit_nc` packed tighter still, and both are since retired, see [REJECTED.md](REJECTED.md)) composes with LMCache in the lab, but its persisted L2 SSD tier is not bit-faithful, so this profile stays fp8-only.

- It builds and runs. The clean TQ image already ships lmcache 0.5.1; graft the format-10 `c_ops.so` from the [fmt10 build](../patches/lmcache-0.5.1-format10-NL_X_NB_NH_BS_TWO_HS.patch) (identical ABI, single file, no recompile). It launches, composes with MTP, stores land (0 format-10 errors), and the L2 SSD tier fills.
- The L2 reload corrupts long-context retrieval. After a container restart LMCache reloads 16–21K tokens in ~26 ms and the output stays fully coherent, yet planted long-context needles vanish: **7/7 miss** across two needles, while a fresh prefill retrieves every time, and the sidecar log confirms LMCache served the reload. Root cause: the format-10 transfer kernel copies bytes for the standard `[NB, NH, BS, 2·HS=512]` fp8/bf16 layout, but k8v4 packs `[…, 262]` (8-bit K + 4-bit V + scales), and the stride mismatch corrupts the L2 serialization round-trip. Coherent but lossy is the failure mode that breaks long-context coding.
- `engine_driven` does not fix it. The `engine_driven` transfer mode ([LMCache PR #4073](https://github.com/LMCache/LMCache/pull/4073)) reclaims the ~900 MB sidecar VRAM (1,370 → 498 MiB), but its SHM-registration handshake is unstable grafted onto the older-nightly TQ base (300 s `register_kv_caches` timeout), and even the prior working run was parked as unstable (≥30K-prefill OOM) with fidelity unverified.
- Conclusion: LMCache's persistence tier only round-trips faithfully with fp8 KV. k8v4 keeps vLLM's in-pool `--enable-prefix-caching` for fast in-pool reuse but has no tiered persistence. A faithful k8v4 tier would need a new lmcache `KVFormatSpec` plus a transfer kernel for the 262-wide packed layout (or forcing opaque BINARY blocks), which is not worth it for a single user.
