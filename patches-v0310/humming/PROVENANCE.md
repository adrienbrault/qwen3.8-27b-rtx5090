# Humming experiment dependency

The official `humming-kernels` 0.1.18 release was uploaded to PyPI at 2026-10-21T03:28:43Z. `Dockerfile.0118` installs its x86_64 wheel by exact URL and SHA256 from `requirements-0118.txt`, without changing dependencies. This is the announcement-day package comparison. The metadata-only backport below is an intermediate diagnostic, not a claim about 0.1.18 performance.

`patches/0001-deviceinfo-compile.patch` backports only `humming/device.py` from vllm-project/humming commit `c00b2e645a4b5ef57174f40254968a2996944217` (2026-10-16). Source: <https://github.com/vllm-project/humming>, Apache-2.0. The original package file matches commit `a74973b5079e42ef861720b62f847ce9d33447f5` exactly. The Dockerfile verifies the complete original and patched file hashes. Humming's GEMM kernels and the installed 0.1.16 package version are otherwise unchanged.

- [PR #148](https://github.com/vllm-project/humming/pull/148), Jinzhen Lin:   prevent recursion when copying or inspecting DeviceInfo.
- [PR #149](https://github.com/vllm-project/humming/pull/149), Michael Goin:   cache one DeviceInfo per CUDA device so torch.compile reads an existing   metadata object, and preserve device identity through copy/deepcopy/pickle.

The first unmodified-package boot selected Humming and failed during Dynamo tracing with the DeviceInfo recursion on 2026-10-21T07:19:51Z; results directory `2026-10-21-071624-r27-humming-screen`. No Humming throughput was measured on that attempt. The CPU test reproduces missing-slot introspection on the real class and then checks copy/deepcopy and a fullgraph Dynamo forward after the patch. It substitutes CUDA metadata only; a real model boot is still required.
