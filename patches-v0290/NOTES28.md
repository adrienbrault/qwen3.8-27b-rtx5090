# R190 / Brief28 — acceptance recalibration investigation

**Status: partial deliverable. Corpus building and acceptance measurement are implemented. Eager TP1 target-feature capture is implemented but untested on GPU. Re-quantization, drafter-forward calibration, and checkpoint export are NOT implemented.** `recalibrate_drafter.py --stage quantize` exits 2 with a specific blocker. It never emits a misleading checkpoint. Phase B is design only.

This brief explicitly requests scripts, no patch. There is no diff, Dockerfile, serving env knob, or PRS marker; no existing vLLM function changed. Scratch source patch application is consequently inapplicable. No network, torch/vLLM imports, or GPU execution was used during development. The missing compressor integration is a limitation of this deliverable and the available evidence, **not proof that recalibration is impossible**.

## Source findings and choice of teacher

Read: `qwen3_dflash.py`, `qwen3_dflash2.py`, GPU `dflash/speculator.py`, `dflash/utils.py`, `eagle/eagle3_utils.py`, `qwen3_5.py`, `qwen3_next.py`, `interfaces.py`, `v1/spec_decode/metrics.py`, `entrypoints/llm.py`, and `worker_base.py` in the supplied patched tree.

* `get_eagle3_aux_layers_from_config` translates target layer IDs [5,19,33,47,61] to aux indices [6,20,34,48,62]. `Qwen3NextModel.forward` calls `_maybe_add_hidden_state` after each decoder, with index `layer_idx+1`. Features are **hidden+residual**, before final model normalization. Five BF16 vectors concatenate to width 25600.
* `DFlashSpeculator.propose` calls `combine_hidden_states` (the 25600→5120 `fc`). The projected context is RMS-normalized and passed through each layer's K/V projections, K norm and RoPE, then inserted into the draft KV cache. Query tokens separately traverse the five draft blocks. The context K/V path uses fused weight buffers; ordinary hooks on the draft's QKV module miss its context activations.
* DFlash2 adds two grouped convolution units per decoder (attention and MLP), their unquantized kernel projections, and the candidate selector's two vocabulary codebooks and hidden projection. Its noncausal attention and context/query visibility must be reproduced. A standard Qwen3 causal-LM forward is not a calibration forward for this model.
* Source uses **1+num_speculative_tokens=10** query rows per request under ns9: anchor plus nine masks (248070). Grouped conv boundaries also use 10. Checkpoint block_size=8 must remain metadata; using eight query slots would calibrate a different runtime shape.
* `load_dflash_model` can share missing embed/head weights with the target. `load_weights` prepends `model.` to non-head checkpoint names and stacks q/k/v and gate/up projections. A generic HF export with already-prefixed names risks double prefixes. Missing embed/head weights, separate mask embedding, and `has_own_*` flags require checkpoint inspection; the JSON alone does not establish ownership.

The OpenAI-compatible server cannot return these chosen hidden layers. `LLM.apply_model` can install hooks inside a local worker. Chosen implementation: separate **eager vLLM TP1 capture process**, native NVFP4 target arithmetic, one GPU, no drafter resident. Hook each selected decoder, immediately copy the residual sum to CPU, assemble a chunk and write it to disk. Arming occurs after model load/warmup; prefix reuse is disabled. `finish` requires positions to equal exactly `range(prompt_tokens)` across chunks, rejecting padding, extra decode rows or incomplete capture. Text MRoPE's three identical axes are supported; multimodal positions fail.

Transformers/llm-compressor source, installed package inventories, upstream DFlash2 training code, and BF16 checkpoint tensor headers are absent here. We cannot establish Transformers' native NVFP4 SM120 execution or offload semantics from this dump. Dequantizing NVFP4 weights to BF16 is **not** equivalent to deployed W4A4 execution: it removes activation quantization and changes GEMM arithmetic. It must not silently be used as this teacher. A 54 GB target plus loader copies also risks exhausting 64 GB host RAM.

The hook captures **TP1 eager prefill**, not TP2 graph-captured verify rows or generated continuation states. Its features preserve NVFP4 arithmetic but may differ through reduction order, prefill kernels, and attention cache behavior. Compare feature error and acceptance against actual served TP2/ns9 traces before using it for final calibration. One-GPU capacity is a budget hypothesis, not a measured fit guarantee.

## Running the delivered stages

Run corpus/measurement in the supplied served container. Run capture in a separate instance of the **same patched image**, with GPU 0 free, no serving process occupying it, local model/corpus/output mounts, and Python scripts importable. These commands execute **inside** that container (for example, via `docker exec CONTAINER ...`). No package installation or model download is performed by the scripts.

Dependencies and version evidence:

| Stage | Packages / versions |
|---|---|
| Dry-runs, preflight, acceptance | Python >=3.10 standard library only; no pip packages |
| Actual chat templating | `transformers==5.15.0` is recorded by drafter config; use the target's local tokenizer and installed compatible tokenizer dependencies |
| Target capture | supplied `vllm==0.29.0rc2` patched 0101..0138 image, `flashinfer-python==0.6.16` per COMMON; CUDA 13, supplied compatible torch/triton builds |
| Intended W4A16 output | `compressed-tensors==0.17.0` format version recorded in checkpoint |
| Missing quantizer stage | llm-compressor version **unresolved**; no tested version/API combination is supplied or claimed |

Do not install a guessed llm-compressor release over the serving stack. Neither exact torch/triton versions nor a compatible compressor pin can be recovered from the evidence. Before any live experiment, record the image digest and package inventory:

```sh
python -m pip freeze > /output/r190-28-packages.txt
python build_calib_corpus.py /sources/calibration --tokenizer /target --count 1024 \
  --min-tokens 1024 --max-tokens 4096 --out /output/calib.jsonl
# Optional generations: add --url http://localhost:8029 --model qwen3.8-27b.
# OPENAI_API_KEY is read from the environment if present.
python recalibrate_drafter.py --stage preflight --bf16-drafter /bf16-draft
CUDA_VISIBLE_DEVICES=0 VLLM_SM12X_PCIE_IPC_AR=0 python recalibrate_drafter.py \
  --stage capture --target /target --corpus /output/calib.jsonl --out /output/features
python measure_acceptance.py --url http://localhost:8029 --model qwen3.8-27b \
  --corpus /output/heldout.jsonl --exclusive --conc 1 --per-kind 12 --runs 3 \
  --out /output/acceptance-base-c1.jsonl
```

If Prometheus uses the checkpoint path rather than API alias, pass that exact label with `--metrics-model`. Output paths must be new; capture artifacts from a failed run are partial and must not be consumed. `.pt` features contain no model weights. The manifest has token IDs, per-chunk positions, corpus/config hashes, target path and vLLM version. Archive the target weight-file checksums separately. No capture directory can be substituted for `/draft`.

Corpus contract: local code extensions include repository files, `.diff` and `.patch`; prose is `.txt`/`.md`. JSON/JSONL records containing both `messages` and `tools` are tool transcripts and must contain tool results. Unrelated JSON objects are ignored; malformed JSON fails. Supply real tool-call IDs, JSON argument strings, results, and function schemas. Templates are rendered with `apply_chat_template(..., tools=..., tokenize=False)` and counted using `add_special_tokens=False`, avoiding duplicate special tokens. The local tokenizer/template is authoritative; no hand-written Qwen template is substituted. Code/prose source slices may cut syntactic units; inspect samples. Tool messages are never truncated. Tool transcripts must already meet the length range. Too few unique eligible transcripts fail instead of duplicating records to pretend there is more data. Maximum file size is 8 MB. Hidden paths, symlinks and binary text are skipped. Use curated directories; this is not a secrets scanner.

Sampling is balanced code/tool/prose (approximately thirds), deterministic by seed without an endpoint, hash-deduplicated, 1024 sequences by default. Optional endpoint generations append an assistant message; generation is capped with reserved template space. No actual tools execute. These calls incur normal generation cost. Generated outputs are not deterministic merely because local sampling is seeded. JSONL includes both rendered text and messages/tools for evaluation. Dry-run outputs are marked and rejected by live stages.

Create held-out data from **disjoint repository/document/transcript sources**, not another random slice of the same sources. Build it separately with the same script. Preserve tokenization, prompts, seed, temperature and generation length between A/B. The supplied `decode_ss.py` implements only code and prose, despite the brief's three-kind wording. This probe adds actual local tool transcripts; it does not claim exact prompt identity with decode_ss's synthetic tasks.

## Missing recalibration work / concrete completion criteria

The supplied quantization scheme is copied without alteration in `drafter-reference.json`: compressed-tensors pack-quantized, all Linear weights, symmetric integer 4-bit, group 128, static group strategy, actorder=null, no input/output activation quantization, ignore `re:.*kernel_projection$`, `re:.*candidate_selector.*`, `re:.*hidden_projection$`. `memoryless_minmax` specifies an observer, **not evidence that the author used GPTQ or AWQ**.

To complete Phase A, obtain and audit the original DFlash2 BF16 model implementation and weight layout, then implement a compressor-compatible model/calibration runner. It must consume the captured feature chunks plus token IDs, sample context/anchor boundaries without future leakage, use ten query rows, and feed both context and query activation distributions into K/V calibration. It must run grouped conv, correct RoPE/sliding noncausal visibility, mask embedding and selector behavior. Verify dense outputs against the served vLLM BF16 drafter on identical inputs before running GPTQ. Do not run `AutoModelForCausalLM` using the config's `model_type=qwen3` and assume architecture dispatch selects DFlash2.

Apply llm-compressor GPTQ to the specified groups, stream calibration batches (do not load all feature files into RAM), and export the original checkpoint naming convention. Inspect exported safetensors, quantization status, scales/group sizes, ignored tensors and optional shared tensors. Compare dequantized weights and dense-versus-quantized draft outputs. Finally load with the unchanged speculative JSON except its model mount, and require ns9, TP2 and the existing DFlash CUDA graph proof in launch-daily.sh. This adapter, verified package pin, and export round-trip are required deliverables still missing here. There is no adapter placeholder that pretends to do this work.

## Measurement and decision

`measure_acceptance` reads exact Prometheus `_total` series from `SpecDecodingProm`, filters `model_name`, preserves engine/position labels, rejects missing/duplicate/nonfinite series, changed label sets and negative deltas, then sums engine deltas. It waits for idle/stable counters before and after each isolated content-kind batch (12 seconds default; increase above the actual metrics publication interval). `--exclusive` is an operator assertion; the client cannot detect all unrelated traffic or a reset that overtakes the old count.

Report accepted tokens / **num_drafts**, not proposed tokens/9; this handles shortened proposals. A draft count is a **request-level** speculative verification event, not a global GPU scheduler iteration at c8/c16. Mean acceptance length is conventionally `1 + accepted/drafts`; acceptance fraction is `accepted/proposed`. Per-position counts divided by drafts are survival probabilities, not conditional acceptance. These are whole-batch counters, including ramp-up/down; wall_seconds is not steady-state decode throughput. Tool EOS may shorten responses; usage is retained. Sampling failures abort the experiment.

First number: **paired held-out code acceptance_fraction delta** versus syvai baseline. An absolute +0.02 is the R186 hypothesis, not a result. Repeat BASE/candidate in alternating order at c1/c8/c16 and several context lengths, including long contexts beyond this 4K calibration set. Run the operator's `decode_ss.py` on code/prose with identical seeds/flags, then the supplied R183 profiler scripts for draft GEMM/step durations and the launcher's graph checks. Tool format/argument correctness needs separate scoring; acceptance alone does not establish it.

Promote only if repeated held-out acceptance rises, steady-state throughput improves by at least ~3% with uncertainty excluding zero, and tool correctness/long-context behavior and memory do not regress. The +0.02 hypothesis implies ~4.3% code / ~7.7% prose tokens-per-step gain at fixed draft cost. A larger step duration erasing this gain, zero held-out improvement, or a gain only in calibration data falsifies the practical proposal. Keep the original checkpoint/mount for rollback. Expected gain is unknown; use 0 to +0.02 absolute as a planning scenario for recalibration, 0 to +0.04 for distillation, **not an empirical interval**. Negative gains remain possible.

## Phase B design: count, memory, streaming

Source-derived count, no biases/sinks/separate mask embedding, excluding shared embedding/head:

| Component | Parameters |
|---|---:|
| Five attention projection sets | 262,144,000 |
| Five MLPs | 1,336,934,400 |
| Norm weights | 62,720 |
| Ten grouped convs (base kernels + projections) | 65,740,800 |
| Feature projection | 131,072,000 |
| Selector codebooks + projection | 128,450,560 |
| Core total | **1,924,404,480** |
| Each embedding/head table, 248320×5120 | 1,271,398,400 |
| Total with both owned tables | **4,467,201,280** |

`parameter_ledger` computes this from config. The vLLM allocated zero mask vector adds 5120 elements; a checkpoint's optional mask tensor/extra keys change actual ownership. `--stage preflight` counts actual safetensors headers without loading tensors. Core BF16 weights = 3.584 GiB; each vocabulary table = 2.368 GiB. Do not call this a 1.5B drafter.

Memory ledger (GiB, planning estimates, not measured peaks):

| Device/use | Budget |
|---|---|
| GPU0 target compressed weights | ~14 GB per brief (~13 GiB if decimal); inspect actual allocation |
| GPU0 attention KV | BF16 4K bound: 16 layers × 2 × 4 heads × 256 × 4096 × 2 B = 0.25 GiB; NVFP4 less plus scales/layout; script reserves 0.5 GiB including engine cache needs |
| GPU0 GDN SSM | ~72 MiB for BF16 matrices at one request (48 layers × 48 value heads ×128×128×2), plus conv states, runtime scratch and speculative snapshots if enabled |
| GPU0 captured five features at 4K | 0.195 GiB on CPU; chunk copies temporarily coexist |
| GPU0 transient activations/kernel workspace | provision 2–6 GiB, confirm especially GDN prefill; eager graphs reserve zero |
| GPU0 CUDA/allocator/headroom | 2–4 GiB; rough total ~18–24 GiB, not a fit guarantee |
| GPU1 trainable core BF16 params+BF16 grads | 7.169 GiB |
| GPU1 FP32 Adam first+second moments | 14.338 GiB |
| GPU1 optional FP32 master weights | 7.169 GiB |
| GPU1 full core Adam subtotal | **28.676 GiB with master**, 21.507 without |
| GPU1 frozen embedding/head if both resident | +4.736 GiB |
| GPU1 activation/scratch/headroom, B=1 | reserve 3–8 GiB; checkpoint layers, sample anchors, benchmark actual peak; scales with B and query/context count |

Full Adam core training with master weights and both tables cannot fit on 32 GiB. Full training of both vocabulary tables is worse: ~66.6 GiB optimizer/parameter/grads alone at 16 bytes/parameter. Freeze shared vocabulary tables/selector initially; consider LoRA on fc/attention/MLP or CPU Adam offload, B=1, gradient accumulation and activation checkpointing. Even without master weights, dense training plus tables/activations is marginal. Optimizer offload adds roughly 21.5 GiB host state for the full core; reserve host RAM for loader and transfer buffers. Avoid loading BF16 target weights on host. Treat 8-bit optimizers as another unverified dependency and numerical change, not a guaranteed fit.

Teacher uses deployed NVFP4 arithmetic: an appropriate teacher for deployed behavior, potentially more relevant than BF16. Freeze target and validate actual speculative rejection/sampling; co-adaptation cannot justify modifying target logits. Stream teacher features and next-token targets across devices in a bounded producer/consumer queue, align token positions explicitly, and discard each chunk after use. Four 4K feature chunks consume ~0.78 GiB host RAM. Dense 4K×248320 FP32 logits alone cost 3.789 GiB; compute loss in chunks, or retain top-k probabilities plus omitted mass with an explicitly approximate loss. Storing only top-k without accounting for the tail does not reproduce full KL.

Feature storage is exactly 51200 bytes/token (~51 KB): 100M tokens = 5.12 TB decimal (4.66 TiB). Phase A disk capture at 1024×2048 tokens is ~107.4 GB; it keeps only one chunk's features in host RAM but is not a bounded-disk streaming trainer. Ensure disk capacity; choose a small smoke corpus first. Phase B must stream, not pre-cache the full budget. At measured effective teacher/trainer throughput X tokens/s, 24 hours yields 86400X tokens: 100/500/1000 tok/s implies 8.64/43.2/86.4M tokens/day; 100M takes ~11.6/2.31/1.16 days. These X values are scenarios, not SM120 benchmarks. Start with 1–5M mixed tokens, evaluate held-out acceptance after each tranche, and expand only on positive net throughput evidence.

The original DFlash/DFlash2 paper/training recipe is not included. No external paper, commit, loss, dataset or optimizer recipe was verified offline. The claim that syvai trained against BF16 is from BRIEF28, not independently established. Do not infer that DFlash2's conv/selector recipe matches original DFlash. Obtain the author's exact revision/license and training recipe before implementing distillation.

## Offline verification, proof and provenance

`sh deliver/verify-0028.sh` copies scripts to `.work/0028`, py_compiles them, runs three dry-runs and dependency-free tests (file corpus/template plumbing, metrics aggregation/reset/missing-series behavior, parameter arithmetic/safetensor metadata), and checks invalid CLI and quantization failure. No ML import occurs. It does not exercise real tokenizer rendering, hooks/cloudpickle, NVFP4 kernel support/VRAM, acceptance HTTP behavior, quantization APIs or checkpoint loading. Dry tokenization is explicitly a JSON/whitespace test double, not Qwen.

GPU capture proof (one process, emitted using logger.info_once after hooks install):

`R190-28 CAPTURE active TP1 eager post-layer=[5,19,33,47,61] width=25600`

This proves hook installation, not valid features or a quantized drafter. Require manifest row/position validation and completion output too. There is intentionally no quantization-success proof.

Provenance for THIRD_PARTY.md: scripts are newly written. Feature selection/residual semantics and worker hook entry points are derived from the supplied vLLM 0.29.0rc2 patched source (vLLM project, Apache-2.0; local patches 0101..0138; upstream source commit not provided). No third-party kernel or training implementation is copied. `drafter-reference.json` is copied from user-supplied `evidence/drafter-config.json`; upstream syvai/incoai model revision/license is not established by that file and must be recorded by the operator. Metric names/formulas follow vLLM `v1/spec_decode/metrics.py`; evaluation methodology follows user-supplied `decode_ss.py` and `R186-ANALYSIS.md` §3.8. No external code, paper citation, license or package API is fabricated.
