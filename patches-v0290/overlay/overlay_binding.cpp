// SPDX-License-Identifier: Apache-2.0
// SM12x (RTX 5090 sm_120 / GB10 sm_121) NVFP4 KV-cache store overlay.
//
// vLLM's in-tree writer (csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu, reached
// through torch.ops._C_cache_ops.reshape_and_cache_flash(..., "nvfp4", ...))
// ALWAYS writes the V block scales in the SM100 trtllm-gen 4-token swizzle.
// The FlashInfer FA2 paged NVFP4 reader that serves SM12x reads K *and* V
// scales linearly (include/flashinfer/attention/prefill.cuh,
// page_produce_kv_sf: sf_gmem_offset = page*stride + head*stride +
// entry_idx*stride_n + col). Swizzled V scales therefore attach the wrong scale
// to most V groups -- fluent output, wrong long-context recall, no crash.
//
// We cannot re-register _C_cache_ops::reshape_and_cache_flash (duplicate def),
// so the PATCHED writer (0002-nvfp4-writer-linear-vscale-sm12x.diff: linear V
// scale when device major >= 12) is compiled here and exposed under a fresh
// namespace: torch.ops.vllm_sm12x.reshape_and_cache_nvfp4(...).
// The vLLM side (0002b) routes NVFP4 stores to it when use_fa2_nvfp4_kv.
//
// Lineage: drowzeys (GB10 recipe, "Agent 2 writer overlay"); adapted for sm_120a.

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

#include "core/registration.h"

#include <string>

// Defined in nvfp4_kv_cache_kernels.cu (the overlay's patched copy).
void reshape_and_cache_nvfp4_dispatch(
    torch::stable::Tensor& key, torch::stable::Tensor& value,
    torch::stable::Tensor& key_cache, torch::stable::Tensor& value_cache,
    torch::stable::Tensor& slot_mapping, torch::stable::Tensor& k_scale,
    torch::stable::Tensor& v_scale, const std::string& kv_cache_dtype);

// Same argument order as reshape_and_cache_flash so callers only swap the op.
void reshape_and_cache_nvfp4(torch::stable::Tensor key,
                             torch::stable::Tensor value,
                             torch::stable::Tensor key_cache,
                             torch::stable::Tensor value_cache,
                             torch::stable::Tensor slot_mapping,
                             std::string kv_cache_dtype,
                             torch::stable::Tensor k_scale,
                             torch::stable::Tensor v_scale) {
  reshape_and_cache_nvfp4_dispatch(key, value, key_cache, value_cache,
                                   slot_mapping, k_scale, v_scale,
                                   kv_cache_dtype);
}

STABLE_TORCH_LIBRARY_FRAGMENT(vllm_sm12x, ops) {
  ops.def(
      "reshape_and_cache_nvfp4(Tensor key, Tensor value,"
      "                        Tensor! key_cache, Tensor! value_cache,"
      "                        Tensor slot_mapping, str kv_cache_dtype,"
      "                        Tensor k_scale, Tensor v_scale) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(vllm_sm12x, CUDA, ops) {
  ops.impl("reshape_and_cache_nvfp4", TORCH_BOX(&reshape_and_cache_nvfp4));
}
