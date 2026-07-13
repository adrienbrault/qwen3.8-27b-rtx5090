FROM vllm/vllm-openai:nightly
USER root

# TurboQuant KV + MTP spec-decode on Blackwell (sm120). Three stacked fixes:
#   1) vllm PR #40914 — route K+1 spec-verify batches through the TQ decode kernel
#      (fixes #40880 degenerate loops).
#   2) fix_spec_output.py — PR #40914 returns the kernel tensor instead of writing
#      the out-param buffer; under FULL_AND_PIECEWISE cudagraphs the return value is
#      discarded, so attention output stayed stale -> constant "!!!!" tokens.
#   3) tq_auto_fallback.py — the MTP/draft runner never inherits cache_config's kv
#      dtype (arrives as "auto"); fall back to $VLLM_TQ_PRESET.
COPY vllm-only.diff /tmp/vllm-only.diff
COPY tq_auto_fallback.py /tmp/tq_auto_fallback.py
COPY fix_spec_output.py /tmp/fix_spec_output.py
COPY fix_spec_guard.py /tmp/fix_spec_guard.py

RUN cd /usr/local/lib/python3.12/dist-packages \
 && patch -p1 --forward --fuzz=3 < /tmp/vllm-only.diff \
 && python3 /tmp/tq_auto_fallback.py \
 && python3 /tmp/fix_spec_output.py && python3 /tmp/fix_spec_guard.py \
 && python3 -c "import ast; ast.parse(open('vllm/v1/attention/backends/turboquant_attn.py').read()); ast.parse(open('vllm/model_executor/layers/quantization/turboquant/config.py').read()); print('ALL_PATCHES_OK')"
COPY tq_splits.py /tmp/tq_splits.py
RUN python3 /tmp/tq_splits.py && python3 -c "import ast; ast.parse(open('/usr/local/lib/python3.12/dist-packages/vllm/config/attention.py').read())"
