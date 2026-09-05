// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 by FlashInfer team.
// Derivative of FlashInfer df8b5c1 ipc_tp2_remote_push_kernel (Apache-2.0).
// Included by bindings.cu after Handle; deliberately not a second translation unit.
#include "flashinfer/comm/pcie_ipc_all_reduce.cuh"

__device__ __forceinline__ float pcie_norm_w2f(float x) { return x; }
__device__ __forceinline__ float pcie_norm_w2f(nv_bfloat16 x) { return __bfloat162float(x); }

template <typename W>
__global__ void pcie_norm_kernel(fi::IpcTp2RemotePushData<nv_bfloat16> p,
                                const nv_bfloat16* residual, const W* weight,
                                nv_bfloat16* norm, nv_bfloat16* resout,
                                int rows, float eps) {
  constexpr int H = 5120, P = H / 8, T = 256;
  const int t = threadIdx.x, peer = p.rank ^ 1;
  int* epoch_slot = p.epoch_slots + blockIdx.x;
  int epoch = fi::advance_scratch_epoch(p.scratch_state, epoch_slot);
  int stage = epoch * 2 * p.rank_stride_packs;
  auto* remote = reinterpret_cast<uint4*>(p.tmp_ptrs[peer]);
  auto* local = reinterpret_cast<uint4*>(p.tmp_ptrs[p.rank]);
  const auto* input = reinterpret_cast<const uint4*>(p.input);
  __shared__ float sums[T];
  // Each CTA owns complete rows on both ranks, avoiding cross-CTA norm barriers.
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    float values[3][8];
    float sum = 0.0f;
    for (int j = 0; j < 3; ++j) {
      int col = t + j * T;
      if (col >= P) continue;
      int idx = row * P + col;
      uint4 own = input[idx];
      fi::store_u4_volatile(remote, stage + p.rank * p.num_packs + idx,
                           fi::clear_pos_zero_u4_16(own));
      uint4 other;
      do {
        other = fi::load_u4_volatile(local, stage + peer * p.num_packs + idx);
      } while (fi::has_pos_zero_u4_16(other));
      uint4 reduced = fi::packed_add_u4<nv_bfloat16>(own, other);
      fi::store_u4_volatile(local, stage + peer * p.num_packs + idx,
                           uint4{0, 0, 0, 0});
      auto* bf = reinterpret_cast<nv_bfloat16*>(&reduced);
      for (int k = 0; k < 8; ++k) {
        // Preserve AR bf16 rounding, but NOT residual bf16 rounding in variance.
        float v = __bfloat162float(bf[k]) + __bfloat162float(residual[idx * 8 + k]);
        values[j][k] = v;
        sum = __fadd_rn(sum, __fmul_rn(v, v));
        resout[idx * 8 + k] = __float2bfloat16_rn(v);
      }
    }
    sums[t] = sum;
    __syncthreads();
    for (int d = T / 2; d; d /= 2) {
      if (t < d) sums[t] = __fadd_rn(sums[t], sums[t + d]);
      __syncthreads();
    }
    float inv = rsqrtf(__fadd_rn(sums[0] / float(H), eps));
    for (int j = 0; j < 3; ++j) {
      int col = t + j * T;
      if (col >= P) continue;
      for (int k = 0; k < 8; ++k) {
        int h = col * 8 + k;
        float gamma = __fadd_rn(pcie_norm_w2f(weight[h]), 1.0f);
        norm[row * H + h] = __float2bfloat16_rn(
            __fmul_rn(__fmul_rn(values[j][k], inv), gamma));
      }
    }
    __syncthreads();
  }
  fi::debug_commit_per_block_epoch(epoch_slot, epoch);
}

std::tuple<at::Tensor, at::Tensor> fused_norm(
    int64_t handle, const at::Tensor& inp, const at::Tensor& residual,
    const at::Tensor& weight, double eps) {
  TORCH_CHECK(handle, "destroyed PCIe IPC workspace");
  const auto* h = reinterpret_cast<const Handle*>(handle);
  TORCH_CHECK(inp.is_cuda() && inp.get_device() == h->device &&
              residual.device() == inp.device() && weight.device() == inp.device(),
              "fused norm device mismatch");
  TORCH_CHECK(inp.scalar_type() == at::kBFloat16 && residual.scalar_type() == at::kBFloat16 &&
              inp.dim() == 2 && inp.size(1) == 5120 && inp.size(0) > 0 &&
              inp.numel() <= h->max_numel && residual.sizes() == inp.sizes() &&
              inp.is_contiguous() && residual.is_contiguous() && weight.is_contiguous() &&
              weight.dim() == 1 && weight.numel() == 5120 &&
              (weight.scalar_type() == at::kFloat || weight.scalar_type() == at::kBFloat16) &&
              eps > 0 && eps < 1, "unsupported PCIe fused norm arguments");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(inp.data_ptr()) % 16 == 0,
              "PCIe fused norm input requires 16-byte alignment");
  c10::cuda::CUDAGuard guard(inp.device());
  auto out = at::empty_like(inp), res = at::empty_like(residual);
  fi::IpcTp2RemotePushData<nv_bfloat16> p{};
  p.tmp_ptrs[0] = h->views.block[0]; p.tmp_ptrs[1] = h->views.block[1];
  p.input = reinterpret_cast<const nv_bfloat16*>(inp.data_ptr());
  p.epoch_slots = h->views.self_signal;
  p.scratch_state = h->views.self_signal +
      fi::scratch_state_offset(kMaxBlocks, 2, fi::ScratchRegion::kBlock);
  p.num_packs = inp.numel() / 8; p.rank_stride_packs = h->max_numel / 8;
  p.rank = h->rank;
  auto stream = c10::cuda::getCurrentCUDAStream(h->device).stream();
  int blocks = std::min<int64_t>(inp.size(0), kMaxBlocks);
#define LAUNCH_NORM(W) pcie_norm_kernel<W><<<blocks, 256, 0, stream>>>(p, \
    reinterpret_cast<const nv_bfloat16*>(residual.data_ptr()), \
    reinterpret_cast<const W*>(weight.data_ptr()), \
    reinterpret_cast<nv_bfloat16*>(out.data_ptr()), \
    reinterpret_cast<nv_bfloat16*>(res.data_ptr()), inp.size(0), float(eps))
  if (weight.scalar_type() == at::kFloat) { LAUNCH_NORM(float); }
  else { LAUNCH_NORM(nv_bfloat16); }
#undef LAUNCH_NORM
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, res};
}
