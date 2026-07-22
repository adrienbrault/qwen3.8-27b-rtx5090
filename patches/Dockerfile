# Pinned to the exact nightly digest the validated profile was built and benchmarked on
# (2026-07 pull). Bumping this digest invalidates every patch below until re-verified —
# they apply with --fuzz=0 (exact context match) precisely so that a moved anchor fails
# the build loudly instead of applying approximately.
FROM vllm/vllm-openai:nightly@sha256:f2ac337893360e23462249c3897e1c92dc8ba8a1ebbc3dd15c4ac931e957438e
USER root

LABEL org.opencontainers.image.source="https://github.com/adrienbrault/qwen3.6-27b-rtx5090" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      org.opencontainers.image.base.name="vllm/vllm-openai:nightly" \
      org.opencontainers.image.base.digest="sha256:f2ac337893360e23462249c3897e1c92dc8ba8a1ebbc3dd15c4ac931e957438e" \
      ai.model.checkpoint="natfii/Qwen3.6-27B-VLM-NVFP4-MTP@2e46c0ed7606f35e357bc5674d20c710fc51b178"

# Image for the Qwen3.6-27B RTX 5090 daily. The CURRENT daily (natfii NVFP4 W4A4
# + fp8 KV + FlashInfer + MTP ns=4) uses patches #5 (PR #42603), FlashInfer 0.6.15,
# and #4 (PR #44993). Patches #1-3 + tq_splits are the TurboQuant 4-bit-KV stack the
# repo shipped as an earlier daily (see docs/HISTORY.md) — inert on the fp8 path, kept
# so the image also serves those configs.
#
# TurboQuant KV + MTP spec-decode on Blackwell (sm120). Stacked fixes:
#   1) vllm PR #40914 — route K+1 spec-verify batches through the TQ decode kernel
#      (fixes #40880 degenerate loops).
#   2) fix_spec_output.py — PR #40914 returns the kernel tensor instead of writing
#      the out-param buffer; under FULL_AND_PIECEWISE cudagraphs the return value is
#      discarded, so attention output stayed stale -> constant "!!!!" tokens.
#   3) tq_auto_fallback.py — the MTP/draft runner never inherits cache_config's kv
#      dtype (arrives as "auto"); fall back to $VLLM_TQ_PRESET.
#   4) so_reasoning_44993.diff — vllm PR #44993 (open, approved; stacked on merged
#      #44297): structured output (response_format json_schema) with a reasoning
#      model returned EMPTY content — the schema JSON leaked into reasoning_content
#      because should_advance's delta window skips </think> when MTP rejects drafts.
#      Needs the launch flag --structured-outputs-config '{...reasoning_parser:qwen3,
#      enable_in_reasoning:false}'. Lifted tool-eval 85->89.
COPY vllm-only.diff /tmp/vllm-only.diff
COPY tq_auto_fallback.py /tmp/tq_auto_fallback.py
COPY fix_spec_output.py /tmp/fix_spec_output.py
COPY fix_spec_guard.py /tmp/fix_spec_guard.py
COPY so_reasoning_44993.diff /tmp/so_reasoning_44993.diff
#   5) install_pr42603_sync.py — from vllm PR #42603 (CLOSED UNMERGED upstream:
#      maintainers rejected a forced sync pending a deeper root cause — treat this
#      as a locally validated WORKAROUND, not a proven fix). The observed race:
#      the MTP draft loop writes shared cudagraph buffers (self.input_ids /
#      self.hidden_states) then launches the draft forward that reads them —
#      under concurrency x K>=2 with fp8 KV on Blackwell this IMA-crashes
#      (upstream #40756 / #35288; "disappears with CUDA_LAUNCH_BLOCKING=1").
#      Adds one current-stream sync after the writes, before the forward.
#      Empirically eliminates the crash on THIS profile: validated across
#      concurrent c4/c8 + deep pp90000 x c4 + 69x2 tool-eval, perf-neutral.
#      (Superseded the R41 #46574 draft-token sanitizer, which never fired.)
COPY install_pr42603_sync.py /tmp/install_pr42603_sync.py

RUN cd /usr/local/lib/python3.12/dist-packages \
 && python3 -c "import vllm; print('BASE VLLM', vllm.__version__)" \
 && patch -p1 --forward --fuzz=0 < /tmp/vllm-only.diff \
 && patch -p1 --forward --fuzz=0 < /tmp/so_reasoning_44993.diff \
 && python3 /tmp/tq_auto_fallback.py \
 && python3 /tmp/fix_spec_output.py && python3 /tmp/fix_spec_guard.py \
 && python3 /tmp/install_pr42603_sync.py \
 && python3 -c "import ast; ast.parse(open('vllm/v1/attention/backends/turboquant_attn.py').read()); ast.parse(open('vllm/model_executor/layers/quantization/turboquant/config.py').read()); ast.parse(open('vllm/v1/structured_output/__init__.py').read()); ast.parse(open('vllm/v1/core/sched/scheduler.py').read()); ast.parse(open('vllm/v1/worker/gpu_model_runner.py').read()); ast.parse(open('vllm/v1/spec_decode/llm_base_proposer.py').read()); print('ALL_PATCHES_OK')"
COPY tq_splits.py /tmp/tq_splits.py
RUN python3 /tmp/tq_splits.py && python3 -c "import ast; ast.parse(open('/usr/local/lib/python3.12/dist-packages/vllm/config/attention.py').read())"

# FlashInfer 0.6.15 (sm120 GDN/TMA fixes). cu130 cubin/jit-cache not published for
# .15, so drop the mismatched 0.6.13 AOT caches and let 0.6.15 JIT-compile at runtime.
# Used by the AR-INT4 + fp8 + MTP-K4 daily (VLLM_ATTENTION_BACKEND=FLASHINFER).
RUN pip uninstall -y flashinfer-cubin flashinfer-jit-cache \
 && pip install --no-cache-dir flashinfer-python==0.6.15 \
 && python3 -c "import flashinfer; print('FLASHINFER', flashinfer.__version__)"
