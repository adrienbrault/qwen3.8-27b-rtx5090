# BRIEF30 — admit the MTP drafter to the PCIe IPC all-reduce (patch 0148)
Read COMMON.md first (dossier layout, deliverable contract, knob/proof-line conventions, no torch imports, never edit src/ in place).

## Why
Patch 0138 (evidence/0138-pcie-ipc-all-reduce-v0290.diff, already applied in src/) gives the served TP2 daily its all-reduce
(FlashInfer-derived PCIe IPC kernel, package src/pcie_ipc_ar21/). Its admission check
(src/vllm/distributed/device_communicators/pcie_ipc_all_reduce.py::_disabled_reason, last clause) refuses every speculative
method except "dflash":

    spec = config.speculative_config
    if spec is not None and spec.method != "dflash":
        return "only the sequential DFlash drafter is supported"

evidence/m4-mtp-engine-log-excerpt.txt shows the consequence on the R197 ladder: with `--speculative-config method=qwen3_5_mtp`
(normalised to "mtp") the engine logs `PCIe IPC all-reduce disabled: rank prerequisites [('only the sequential DFlash drafter is
supported', ...)]` and falls back to ['CUSTOM','PYNCCL']. The operator needs an image where the MTP drafter (the checkpoint's own
MTP head, Qwen3.8-27B, hidden 5120, TP2 bf16) runs WITH the PCIe IPC all-reduce, so MTP-vs-DFlash is measured on the same
all-reduce as the daily. NOTES21.md (evidence/) says the restriction exists because the R185 dump proved only "no drafter" and
DFlash: DFlash drafts sequentially on the forward stream under the same workspace, one drafter graph replay per target step.

## The MTP drafter on this source (read these first)
- src/vllm/v1/worker/gpu/spec_decode/__init__.py — method "mtp" → MTPSpeculator.
- src/vllm/v1/worker/gpu/spec_decode/mtp/speculator.py — MTPSpeculator(AutoRegressiveSpeculator): load_eagle_model, skip_topk hooks.
- src/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py — capture() captures TWO graph families through
  SpeculatorCudaGraphManager (src/.../autoregressive/cudagraph_utils.py): `prefill_cudagraph_manager.capture(self._prefill, ...)`
  and, when num_speculative_steps > 1, `decode_cudagraph_manager.capture(self._generate_draft | _generate_fused_drafts, ...)`.
  The drafter has no stream of its own (current_stream()). Rows per drafter forward = num_reqs (SEQS 16 on the daily → ≤ 16, far
  below max_rows=320) for decode; prefill rows = tokens of the prefill dummy (find the capture sizes both managers use).
  The MTP head is ONE decoder layer of the target (attention block, o_proj + down_proj TP all-reduces at hidden 5120) plus norms;
  the target's lm_head is reused. The decode graph is REPLAYED ns times back-to-back per target step with the same shape (the
  engine warns: "num_speculative_tokens > 1 will run multiple times of forward on same MTP layer").
- DFlash's hooks (the template): src/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py capture() calls
  `pcie_comm.prepare_draft(self.draft_model_config.get_hidden_size(), self.dtype)` then wraps the drafter graph capture in
  `pcie_comm.capture()`; src/vllm/v1/worker/gpu/model_runner.py::capture_model calls `pcie_comm.prepare_for_capture(self.vllm_config)`
  and wraps the target capture in `pcie_comm.capture()`; gpu_worker.py wraps ensure_model_parallel_initialized in
  set_current_vllm_config when VLLM_SM12X_PCIE_IPC_AR=1. Grep `pcie` in src/vllm/v1/worker/gpu/ for every hook.
- The runner's eager dummy propose runs BEFORE capture (model_runner.py, grep get_mtp_target_hidden_states / dummy). State from
  source what the all-reduce does for a shape that has not been prepared yet (DFlash survives that path today, so there is a
  fallback or the shapes are pre-registered — say which, with line refs).

## Deliverable: deliver/0148-pcie-ipc-mtp-drafter-v0290.diff (applies on src/ at fuzz 0, AFTER 0147 — see below)
Knob `VLLM_SM12X_PCIE_IPC_MTP=1` in src/vllm/envs.py (unset ⇒ byte-identical behaviour: today's message and refusal).
1. Admission: with the knob set, `_disabled_reason` also admits `spec.method == "mtp"`. Every other method is still refused with
   the existing message. Keep the exact string "PCIe IPC all-reduce enabled" (the launcher asserts on it) and add ONE
   logger.info_once proof line on admission: `SM12X PCIe IPC: MTP drafter admitted (sequential AutoRegressive drafter, rows<=<n>)`.
2. Shape registration: before the FIRST drafter capture, every row count that either SpeculatorCudaGraphManager will capture
   must be prepared at the drafter's hidden size (5120) and dtype — derive both size lists from the managers/config (decode sizes
   are request counts, prefill sizes are token counts; do NOT assume the target's prepare_for_capture list covers them). Use
   prepare_draft(hidden, dtype) if its 1..MAX_ROWS registration already covers everything (say so), otherwise register exactly.
   NOTES21's hang definition: an unresolved-shape collective inside capture is a boot hang, not a fallback.
3. Capture wrapping: both `prefill_cudagraph_manager.capture(...)` and `decode_cudagraph_manager.capture(...)` in
   autoregressive/speculator.py sit inside `pcie_comm.capture()` (mirror DFlash exactly; no-op when the comm is not active).
4. THE question that decides safety (answer in NOTES30.md with .cuh/.cu line references, from src/pcie_ipc_ar21/): the MTP decode
   graph is replayed ns times back-to-back per target step with one shape, then the target graph replays; DFlash alternates
   drafter/target once per step. NOTES21 lists "graph replay / alternating-size epoch behaviour" as unverified and describes
   2-epoch scratch regions. Is the epoch/flag advanced on DEVICE inside the kernel (repeat replay of one graph is safe for any
   sequence), or chosen HOST-side at capture time (consecutive replays of one captured graph reuse a live epoch → corrupt)? If
   host-side, the fix lives in the kernel/binding and will not fit this brief's budget: say so precisely, deliver items 1–3 anyway
   (the operator will test empirically with a bitwise ruler), and sketch the binding change in NOTES30 (no diff required).
5. Also state whether the MTP drafter's all-reduce call sites go through the same tensor_model_parallel_all_reduce dispatch that
   0138 hooks (device_communicators/cuda_communicator.py) or through anything else (custom op path, compiled region), so the
   admission is actually exercised on every drafter collective.

Base: the image this layers on is `vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616-pcieipc-bsshash` = src/ + 0147
(evidence/0147-bss-not-a-compile-factor-v0290.diff, touches vllm/config/parallel.py only). `deliver/verify-0148.sh` must copy src/
to a scratch tree, apply 0147 then 0148 with `patch -p1 --fuzz=0 --batch --forward`, CHECK EXIT CODES (never grep the output),
py_compile every touched file, and run a control (0148 must also fail cleanly on an unpatched-0138 tree = evidence proves it
targets the 0138 code). Also `deliver/touched-0148.txt` and `deliver/Dockerfile.pcie-mtp` (copy evidence/Dockerfile.bss-not-a-compile-factor
verbatim in shape: `ARG BASE=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616-pcieipc-bsshash`, `--network=none`, patch at fuzz 0,
py_compile touched, marker `/opt/prs-markers/0148`). Patch budget < 250 diff lines. No torch imports at module level in envs.py.

Measurement plan for NOTES30.md (the operator runs it on the R197 ladder, arms M3p/M4p = MTP ns3/ns4 on the new image with
PCIE_IPC=1 vs M3/M4 on CUSTOM): boot proof lines; the bf16 decode ruler at ctx 0 / 30K, which must be BITWISE 20/20 vs the CUSTOM
arm when the compile hash sets are equal and loaded (R185: pcie_ipc vs CUSTOM identical on every paired ruler); probes/decode_ss.py
code c1/c8/c16, prose c1/c8. Predict the direction only (pcie_ipc gained +2–4 % steps/s at DFlash shapes; MTP has ns× more
drafter all-reduces per step at rows = num_reqs, so the relative gain should be larger at c1 and about equal at c16); do not
claim a number.
