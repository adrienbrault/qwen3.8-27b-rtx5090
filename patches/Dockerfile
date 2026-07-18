FROM vllm/vllm-openai:nightly
USER root

# Image for the Qwen3.6-27B RTX 5090 daily. The CURRENT daily (Lorbus INT4-AutoRound
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
#   5) install_pr42603_sync.py — vllm PR #42603: fix the MTP draft-loop buffer
#      race that IMA-crashes under concurrency x K>=2 with fp8 KV on Blackwell
#      (upstream #40756 / #35288). The draft loop writes shared cudagraph buffers
#      (self.input_ids / self.hidden_states) then launches the draft forward that
#      reads them — CUDA-async, so under concurrency the draft FlashInfer kernels
#      observe stale/partial buffers -> illegal memory access ("disappears with
#      CUDA_LAUNCH_BLOCKING=1"). Adds one current-stream sync after the writes,
#      before the forward. Restores MTP K=4 on AR-INT4+fp8+FI; validated crash-
#      free across concurrent c4/c8 + deep pp90000 x c4 + 69x2 tool-eval.
#      (Superseded the R41 #46574 draft-token sanitizer, which never fired — the
#      fault was never an out-of-vocab draft token, it was this buffer race.)
COPY install_pr42603_sync.py /tmp/install_pr42603_sync.py

RUN cd /usr/local/lib/python3.12/dist-packages \
 && patch -p1 --forward --fuzz=3 < /tmp/vllm-only.diff \
 && patch -p1 --forward --fuzz=3 < /tmp/so_reasoning_44993.diff \
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
