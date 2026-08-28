# v0.28.0 V-scale writer verdict

**Verdict: the writer assumption is unchanged. The v0.27 `0002` csrc change and
the runtime overlay apply unchanged to v0.28.0.** SM12x FlashInfer FA2 NVFP4
still requires the linear-V-scale overlay; the stock v0.28.0 writer must not be
used for serving on this route.

Evidence in `refs/nvfp4_kv_cache_kernels-v0280.cu`:

- Lines 34-48 define the SM100 four-token scale swizzle.
- Lines 282-290 write K scales linearly.
- Lines 291-298 send every V scale through `swizzle_scale_offset`; there is no
  device-generation or reader-layout condition.
- Lines 333-337 still impose the swizzle's `block_size % 4 == 0` requirement
  unconditionally, further confirming there is no linear path.
- Lines 388-417 dispatch both `nvfp4` and `nvfp4_4over6` through the same kernel
  signature with no V-layout argument. The `nvfp4_4over6` enum is only a
  store-time scale-search choice (lines 28-32), not a linear-layout variant.

`0102-nvfp4-writer-linear-vscale-sm12x-v0280.diff` is therefore intentionally
the unchanged 75-line csrc patch: it adds `swizzle_v_sf`, sets it false for
device major 12+, writes V scales through the linear branch in that case, and
passes the flag to both scale-search variants. It passed `patch --dry-run -p1`
against the supplied v0.28.0 reference writer.

The files in `overlay/` are byte-for-byte copies of the validated v0.27 overlay.
Their binding still matches the v0.28.0 dispatch signature shown at reference
lines 311-315, so no overlay source change is required. Compilation and numeric
reader/writer validation remain GPU/image-build checks; they could not be run in
this CPU-only work package.
