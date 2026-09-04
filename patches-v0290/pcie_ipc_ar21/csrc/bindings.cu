// SPDX-License-Identifier: Apache-2.0
// R185: PyTorch/CUDA IPC ownership adapter for FlashInfer df8b5c1's header.
// No TVM FFI, FlashInfer headers beyond the vendored kernel, or Python C API.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include <cstring>
#include <memory>
#include <tuple>
#include <vector>

#include "flashinfer/comm/pcie_ipc_all_reduce.cuh"

namespace fi = flashinfer::comm::pcie_ipc;
namespace {
constexpr int kWorldSize = 2;
constexpr int kMaxBlocks = 128;
constexpr int64_t kCapacity = 320LL * 5120;
static_assert(sizeof(void*) == sizeof(int64_t));

struct Handle {
  fi::PeerViews views;
  int rank;
  int device;
  int64_t max_numel;
};

void check_capacity(int64_t max_numel) {
  TORCH_CHECK(max_numel > 0 && max_numel <= kCapacity && max_numel % 8 == 0,
              "invalid PCIe IPC capacity");
}

std::tuple<int64_t, std::vector<int64_t>> allocate(int64_t max_numel, int64_t device) {
  check_capacity(max_numel);
  c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
  const auto bytes = fi::workspace_size(kWorldSize, max_numel, 2, kMaxBlocks);
  void* ptr = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&ptr, bytes));
  try {
    cudaIpcMemHandle_t ipc{};
    C10_CUDA_CHECK(cudaMemset(ptr, 0, bytes));
    C10_CUDA_CHECK(cudaIpcGetMemHandle(&ipc, ptr));
    // Complete initialization before any rank can import this allocation.
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    const auto* data = reinterpret_cast<const unsigned char*>(&ipc);
    std::vector<int64_t> encoded(data, data + sizeof(ipc));
    return {reinterpret_cast<int64_t>(ptr), encoded};
  } catch (...) {
    cudaFree(ptr);
    throw;
  }
}

int64_t open_ipc(const std::vector<int64_t>& encoded, int64_t device) {
  TORCH_CHECK(encoded.size() == sizeof(cudaIpcMemHandle_t), "invalid CUDA IPC handle");
  cudaIpcMemHandle_t ipc{};
  auto* data = reinterpret_cast<unsigned char*>(&ipc);
  for (size_t i = 0; i < encoded.size(); ++i) {
    TORCH_CHECK(encoded[i] >= 0 && encoded[i] <= 255, "invalid IPC handle byte");
    data[i] = static_cast<unsigned char>(encoded[i]);
  }
  c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
  void* ptr = nullptr;
  C10_CUDA_CHECK(cudaIpcOpenMemHandle(&ptr, ipc, cudaIpcMemLazyEnablePeerAccess));
  return reinterpret_cast<int64_t>(ptr);
}

void close_ipc(int64_t ptr, int64_t device) {
  c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
  C10_CUDA_CHECK(cudaIpcCloseMemHandle(reinterpret_cast<void*>(ptr)));
}

void free_local(int64_t ptr, int64_t device) {
  c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
  C10_CUDA_CHECK(cudaFree(reinterpret_cast<void*>(ptr)));
}

int64_t init(const std::vector<int64_t>& ptrs, int64_t rank,
             int64_t max_numel, int64_t device) {
  check_capacity(max_numel);
  TORCH_CHECK(ptrs.size() == kWorldSize && rank >= 0 && rank < kWorldSize,
              "PCIe IPC requires exactly two ranks");
  for (auto ptr : ptrs) TORCH_CHECK(ptr != 0 && ptr % 16 == 0, "invalid slab pointer");
  auto h = std::make_unique<Handle>();
  h->views = fi::make_peer_views(ptrs.data(), kWorldSize, rank,
      fi::compute_workspace_layout(kWorldSize, max_numel, 2, kMaxBlocks));
  h->rank = rank;
  h->device = device;
  h->max_numel = max_numel;
  return reinterpret_cast<int64_t>(h.release());
}

void dispose(int64_t handle) { delete reinterpret_cast<Handle*>(handle); }

at::Tensor all_reduce(int64_t handle, const at::Tensor& inp,
                      int64_t blocks, int64_t threads, int64_t variant) {
  TORCH_CHECK(handle != 0, "destroyed PCIe IPC workspace");
  const auto* h = reinterpret_cast<const Handle*>(handle);
  TORCH_CHECK(inp.is_cuda() && inp.get_device() == h->device,
              "input must be on the workspace device");
  TORCH_CHECK(inp.scalar_type() == at::kBFloat16 && inp.dim() == 2 && inp.is_contiguous(),
              "PCIe IPC expects contiguous 2-D bf16");
  TORCH_CHECK(inp.numel() >= 16 && inp.numel() <= h->max_numel && inp.numel() % 8 == 0,
              "invalid PCIe IPC payload size");
  TORCH_CHECK(blocks > 0 && blocks <= kMaxBlocks && threads >= kWorldSize &&
              threads <= 1024 && threads % 32 == 0 && (variant == 0 || variant == 1),
              "invalid TP2 launch configuration");
  c10::cuda::CUDAGuard guard(inp.device());
  auto source = inp;
  if (reinterpret_cast<uintptr_t>(source.data_ptr()) % 16 != 0) {
    // Pointer alignment is rank-local. Align without changing the collective.
    source = at::empty(inp.sizes(), inp.options());
    source.copy_(inp);
  }
  auto out = at::empty_like(inp);
  const auto stream = c10::cuda::getCurrentCUDAStream(inp.get_device()).stream();
  // Exact upstream launcher, with PDL disabled as required by its binding.
  C10_CUDA_CHECK(fi::all_reduce<nv_bfloat16>(
      reinterpret_cast<const nv_bfloat16*>(source.data_ptr<at::BFloat16>()),
      reinterpret_cast<nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      inp.numel(), h->views, h->rank, kWorldSize, kMaxBlocks, h->max_numel,
      blocks, threads, static_cast<fi::Variant>(variant), false, stream));
  return out;
}
}  // namespace

TORCH_LIBRARY(pcie_ipc_ar21_native, m) {
  m.def("allocate(int max_numel, int device) -> (int, int[])", &allocate);
  m.def("open_ipc(int[] encoded, int device) -> int", &open_ipc);
  m.def("close_ipc(int ptr, int device) -> ()", &close_ipc);
  m.def("free_local(int ptr, int device) -> ()", &free_local);
  m.def("init(int[] ptrs, int rank, int max_numel, int device) -> int", &init);
  m.def("dispose(int handle) -> ()", &dispose);
  m.def("all_reduce(int handle, Tensor inp, int blocks, int threads, int variant) -> Tensor");
}
TORCH_LIBRARY_IMPL(pcie_ipc_ar21_native, CUDA, m) {
  m.impl("all_reduce", &all_reduce);
}
