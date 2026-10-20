#!/usr/bin/env bash
# Served launcher since 2026-10-20 13:53 UTC: the no-block-drop BA4 daily below with a 15.90 GB/card KV pin
#   at 16 sequences, pool 1,490,437 tokens, daily admission band 1,460,000 to 1,525,000.
#   The text-only ladder qualified 15.90 and 16.90 on three boots each; concurrent decode, cold prefill and images
#   left 1,227 MiB free at 15.90 and 99 MiB at 16.90. The 16.02 GB follow-up read 925 and 1,047 MiB on two boots.
#   Evidence: bench/results/r27-kv-pin.md, results 2026-10-20-122821-r27-pin-ladder,
#   2026-10-20-131748-r27-headroom-stress, 2026-10-20-135519-r27-headroom-stress,
#   2026-10-20-135353-pin-promote. Rollback: scripts/serve-v031-nbd-daily.sh.
# Previous launcher, 2026-10-20 (no block drop): the BA4 launcher below with "disable_eagle_block_drop":true in the MTP
#   speculative config (vLLM v0.31.0's option; see the NBD_ block). On a prefix-cache hit vLLM keeps the last full-attention
#   block instead of dropping and re-prefilling it. Flags screen on 2026-10-20 (results 2026-10-20-051649-r27-flags-screen,
#   C E C E): greedy 32K revisits equal the cold text in 12 of 12 prompts (1 of 12 without), warm TTFT 0.071 s at 32K (0.275
#   without); decode within -0.4..+2.3 % (results 2026-10-20-070708-r27-flags-screen). The boot asserts the flag on the container
#   and the UP line says "no-block-drop". Evidence: bench/results/r27-no-block-drop.md.
#   Rollback: scripts/serve-v031-ba4-daily.sh.
# Previous launcher (2026-10-20, r27 BA4): L1a + one Python/Triton layer, image
#   vllm-qwen38:v0310-r27c-A_S-BA4-L1a-ubplan (sha256:1df9b4c4e1d24926104c0fe5fbddce5aba7a74c0e694e54f1747d9d91622d8d7).
#   patches-v0310/ba4/Dockerfile adds L1a_BA4.diff; VLLM_SM12X_BA_M4=1 lets a Triton GEMM serve the 48 GDN in_proj_ba
#   layers (bf16, M=4 x N=48 x K=5120) on the MTP ns3 verify step. Each rank admits the layer from the producer reports and
#   launch binding of the screen's preflight, mounted read-only at /ba4-results; a device, code or weight mismatch logs
#   "SM12X BA4 INACTIVE" and that layer runs the original GEMM. BA4 never fails the boot; the UP line reports n/96 active.
#   Screen on 2026-10-19 (results 2026-10-19-r27-audition-screen-070155-3532152, L1a BA4 BA4 L1a): c1 code +13.0 %,
#   c1 prose +11.2 % (WIN); c4, c8 and prefill TIE or NOISE. Correctness (results 2026-10-19-*-ba4-lean-gate): GEMM error
#   on live decode-verify operands equals the correctly rounded bf16 product's; GSM8K 5-shot T=0 n=200 0.905 vs L1a 0.885;
#   tool-eval 91.0 vs 89.8; greedy text differs from L1a. Evidence: bench/results/r27-ba4.md.
#   Rollback: scripts/serve-v031-l1a-daily.sh.
# Previous launcher (2026-10-17 to 2026-10-20, r27 L1a): A_S + one Python layer, image
#   vllm-qwen38:v0310-r27c-A_S-L1a-ubplan (sha256:631a6f24ccca2f5470a3d0772d833670a16613c69bb4191663f718e473086f5e).
#   patches-v0310/Dockerfile.A_S-L1a-ubplan adds L1a-draft-ub-plan.diff; VLLM_SM12X_DRAFT_UB_PLAN=1 makes the
#   MTP drafter's q_len=1 FlashInfer plan use the CPU sequence-length upper bound instead of a blocking D2H copy.
#   Evidence: bench/results/r27-l1a.md. Rollback: scripts/serve-v031-splitkv-daily.sh.
# Previous launcher (2026-10-16 to 2026-10-17, r27 A_S): the 27B on vLLM v0.31.0 + FlashInfer 0.7.0.post1 with the 0136 split-KV override,
#   image vllm-qwen38:v0310-r27c-A_S-splitkv (sha256:2a081946bc9c0b6694cb246391a91ba19995d7dd55245afabdb282f97cf1808e), built from
#   patches-v0310/: Dockerfile.v0310 (the whole chain rebased onto v0.31.0, see PORT-STATUS.md) plus Dockerfile.A_S-splitkv (the
#   0136 guard lambda accepts FlashInfer 0.7's (kv_data_type, device) signature). Checkpoint, KV pin (14.86 GB, pool 1,391,795),
#   16 sequences, MTP ns3, NVFP4 KV, bf16 GDN state, pcie_ipc all-reduce, batch-sharded sampling and tiers are unchanged.
# Launch delta against serve-r231-nvidia-daily.sh: the image; env VLLM_SM12X_NVFP4_PREFILL_SPLIT_KV=1 and VLLM_USE_V2_MODEL_RUNNER=1;
#   flags --mamba-backend triton --gdn-prefill-backend flashinfer (--mamba-cache-mode align and --enable-prefix-caching come from
#   serve-v0280-daily.sh as before). All four key on the image tag (V0310 below), so a v0.29 image passed as CAND_IMG keeps the
#   v0.29 argv. This is the argv the r27 screen and qualify runs booted, checked offline against their resolved launches.
# Why split-KV: FlashInfer 0.7's prefill plan() forces disable_split_kv for an NVFP4 KV cache on SM120/121
#   (_nvfp4_kv_requires_disabled_split_kv). The MTP verify attention then takes 1,377 µs per decode step instead of 380 and
#   decode at 8 streams runs 4.6-10.2 % slower than the v0.29 launcher (results 2026-10-15-r27-audition-screen-222033-601088,
#   2026-10-16-r27-audition-profile-035416-3692711). With the override: 380 µs and the v0.29 step time (results 2026-10-16-r27-audition-
#   profile-063752-861443). Evidence and gates: bench/results/r27-v031-splitkv.md.
# Rollback: serve-r231-nvidia-daily.sh (the serving host keeps the 2026-10-16 v0.29 launcher as launch-daily-v0290-r207.sh).
# DAILY (2026-09-04 → 2026-10-16, R168 promotion, user "Promote, go"; sheet flan/r168-DECISION.md): the vLLM 0.29 nvfp4-KV route.
#   RedHatAI/Qwen3.8-27B-NVFP4 weights (unchanged since R156) + NVFP4 KV + DFlash2 ns9 draft_tp2 in CUDA graphs on the dual
#   5090s (TP2), syvai W4A16 drafter, native disk tier with 0137 LRU eviction, 16 GiB CPU tier, embed-table UVA offload (0135).
#   Image S = rc2 + patches-v0290 + FlashInfer 0.6.16.post3 swap (`vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616`, split_kv=0).
# Why (r168e/r169/r170/r172/r173, all vs the R156 bf16 rulers): pool 937,795 tokens pinned (fp8 daily 654,491 / 628,798,
#   +43%); disk tier actually SERVES 131K/220K prompts (4/4; the v0.28 daily never served above ~100K, 0/4 even at 16 GiB);
#   tier eviction holds a cap under floods and restarts (the fp8 daily's tier stranded at 100% on 2026-09-01); S image is the
#   closest of the three 0.29 attention paths to bf16 (dense top-1 92.797%, +0.753% PPL; fp8 daily 93.07% / +0.376%), and the
#   bf16 DECODE reference (r173c) puts it in the same 0.0051–0.0062 band as fp8 KV at 30K.
# Cost (r173/r173b, steady-state): code c8 ≈ −5% (1,134–1,146 vs 1,143–1,153), deep30k prose c1 ≈ −5% (147 vs 150–164),
#   short-request admission lower at c32 (nvfp4 blocks are 2,944 tokens). ns7 (+12.5% c8) was RETRACTED on the bf16 decode
#   ruler (median 2× ns9's at 30K) — do not flip NS on the daily without a new r173c-style ruler run.
# Pin: KV pool is PINNED (kv_cache_memory_bytes) because the util path sizes the pool before graph capture (Bug C). 14.5 GB at
#   SEQS 8 leaves 1,311–1,809 MiB free after pre-warm across boots (r173b); the guard below fails the boot under MIN_FREE_MIB.
# Bug B dodge ASSERTED (XQA off, FIWS 512 MiB): nvfp4 prefill above MNBT≈4,929 corrupts under XQA (R155); XQA=0 is the actual
#   fix (R155n13a2 turned the red 262K shape green), and MNBT 4096 < M*=4,929 is additionally on the safe side of that boundary.
# MNBT 8192, and it is NOT a free parameter: block_size is 1,472 and scheduler.py:438-446 snaps every non-final prefill chunk DOWN
#   to a multiple of it (the escape hatch needs `not use_eagle`, and method "mtp" IS eagle-family), so the effective chunk is
#   floor(MNBT/1472) blocks — 8192 -> 5 blocks = 7,360 tokens. Only block multiples mean anything here: 8192 and 7424 are the same
#   engine. MNBT was 4096 (2 blocks) for 76 minutes on 2026-09-08 and R211 reverted it: the R209d mechanism is real (measured stall
#   median ratio 0.40 against 2,944/7,360 = 0.400 predicted) but halving the chunk also roughly DOUBLES the stall rate, so 2 blocks
#   cost -7.5 % output throughput, +2.6 % steady E2EL and +12 % ramp E2EL to buy mean ITL 104 -> 97 ms. Above 5 blocks buys nothing
#   at realistic prefix reuse: the stall is min(chunk, the request's remaining prefill), and at 82 % hit a request has ~8K uncached.
#   TTFT is ~75 % QUEUE in every arm — SEQS, not MNBT, is the lever there (R211 arm E: SEQS 32 = TTFT -34 %, E2EL +12 %).
# EXP=1 → :8029 / vllm-exp / eval-l2 (batteries); EXP=eval → :8030 / vllm-eval; default → :8020 daily. Experiments may pass
#   CAND_IMG / SPLIT_KV / OFFLOAD / KV_BYTES / SEQS / SPEC_NS / SPEC_DTP / TIER_* / CPUB; the daily port ignores the env. Since the
#   promotion, experiments default to the DAILY image and to its spec route (MTP ns3 since R207; knobs off unless PCIE_IPC=1 / BSS=1) — pass
#   CAND_IMG=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs for the 0.6.18 rc2 image, or SPEC_METHOD=dflash SPEC_NS=7 for a DFlash control.
#   Since r27 A_S (2026-10-16) the default image is v0.31 and SPLIT_KV defaults to 1 on it; a v0.29 control is
#   CAND_IMG=vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616-pcieipc-bsshash-mtppcie-mtpcache-eagleshift (SPLIT_KV then defaults to 0).
#   EXP boots still need PCIE_IPC=1 BSS=1 SEQS=16 to match the daily (unchanged rule).
# R182 (2026-09-04, user "Ok promote"): SSM state cached in bf16 (`--mamba-ssm-cache-dtype bfloat16`). The hybrid allocator stores a GDN
#   state snapshot per block (mamba_cache_mode=align) and sizes the attention block to that page, so the fp32 state made a request cost
#   6.4% of the pool at admission + ~0.2%/1K tokens (R178: 15 short or four 100K requests fill 903K "tokens"). bf16 halves the page:
#   block 2,944 → 1,584, pool 1,020,596 at the same 13.98 GB pin (R179), fixed cost 3.5%, 100K prompt 16.5%, five 100K co-resident
#   (R180); rulers vs bf16 neutral (dense 92.771%/+0.744% vs 92.716%/+0.770%), 80-chunk decode ruler at 30K 0.00576 vs 0.00443
#   median |Δlogprob|, 31/31 per chunk, tails equal (R181). Block change = every tier hash changes; native-l2 wiped at promotion.
# R197 (2026-09-05 14:0x UTC, user "I validate switching from d9 to d7"): DFlash draft length 9 → 7. The speculative-length ladder
#   (results/2026-09-05-r197-spec-ladder, :8029, same image/flags/pin, SEQS 16, PCIE_IPC=1, BSS=1) read ns7 vs ns9: code c8 1,633 vs 1,480 t/s
#   (+16 % steps/s), prose c8 1,134 vs 941, code c16 2,455 vs 1,996 (+25 % steps/s), code c1 273 vs 306 (−11 %), prose c1/30K within noise;
#   ns6 ≡ ns7 on tok/s, ns8 between, ns10/11 below ns9 batched. Acceptance per draft token falls with ns (0.47 at 6 → 0.25 at 11).
#   ns IS a compile factor in practice (the attention block follows the speculative slot count: ns7 → 1,552-token block, pool 1,052,277 at
#   the 13.98 GB pin vs 1,584 / 1,020,596 at ns9), so the cross-ns numerics are void under R193 and the tier hashes change: native-l2 is
#   wiped on the first daily boot whose block stamp differs (below). Rollback: launch-daily-r195-ns9-0905.sh.
# R185 (2026-09-05, user "seems like no brainer? Lets use?!"): FlashInfer main's pcie_ipc all-reduce (patch 0138 + package pcie_ipc_ar21;
#   image `...-fi0616-pcieipc` = the S image + one 359 KB layer, Dockerfile.pcieipc) behind VLLM_SM12X_PCIE_IPC_AR=1, ASSERTED at boot
#   ("PCIe IPC all-reduce enabled" + backend order PCIE_IPC, CUSTOM, PYNCCL). R185/R185b (results/2026-09-05-r185-pcieipc, knob on vs
#   off on the same image, same night): code c1 +4.6%, prose c1 +5.4%, prose 30K +3.1%, c16 +2.6%, c8 flat in tok/s = +4.9% steps/s;
#   numerics identical on every paired ruler (decode ctx0/30K, agentic). Experiments: PCIE_IPC=1 opts in (default OFF so batteries stay
#   comparable with the R183 band); the daily port forces it on. Gates on the live daily: r189-promote-pcieipc.sh.
# Rollback: bash /srv/qwen5090/launch-daily-r182-nopcie-0905.sh (fi0616 image, no 0138, frozen pre-R185 launcher; tear this one down
#   first); older: launch-daily-r174-ssm-fp32-0904.sh (fp32 SSM), launch-daily-redhat-fp8-0902.sh (v0.28 fp8 daily)
# R195 (2026-09-05, user "yes" on the R193e sheet): batch-sharded sampling (`--enable-batch-sharded-sampling`: each TP rank samples its
#   half of the batch from local logits, an all-to-all replaces the 19.9 MB per-step logits all-gather). Image `...-pcieipc-bsshash` =
#   the pcieipc image + patch 0147 (the flag removed from ParallelConfig.compute_hash, so ON/OFF share one AOT compile artifact; marker
#   /opt/prs-markers/0147 asserted). R193e (results/2026-09-05-r193e-pin-bss; one artifact + VLLM_TRITON_FORCE_FIRST_CONFIG=1 on both arms):
#   ON vs OFF bitwise 20/20 median 0 at ctx0 and 30K; steps/s +2.6% c8, +4.5% c16, +1.5% c1 (R193b/R193c same size). Seeded T>0 requests
#   draw a different sample stream than the unsharded sampler (R187). Experiments: BSS=1 (or the flag in EXTRA_ARGS_APPEND) opts in;
#   the daily port forces it on and asserts the "Batch-sharded sampling enabled" line. Gates on the live daily: r195-promote-bss.sh.
#   NOTE the daily's compile artifact changed with the image (0147 changes the hash string): cd95c505/d5e217de/fd65aadf (r193b OFF-h),
#   another compile-lottery draw (R193) — its bf16 ruler position is read by r195.
# Rollback: bash /srv/qwen5090/launch-daily-r189-nobss-0905.sh (pcieipc image, no 0147, sharded sampling off; frozen pre-R195 launcher;
#   same block size so the native-l2 tier is shared; tear this one down first).
# R207 (2026-09-06 23:09 UTC boot, 23:39 gates green, user "Alright let's switch to mtp"; sheet flan/r206-DECISION.md): the speculative route switches from the
#   DFlash2 drafter to the checkpoint's own MTP head, ns3. Image `...-bsshash-mtppcie-mtpcache-eagleshift` = the R195 daily image + patch
#   0148 (the sequential MTP drafter admitted on the pcie_ipc all-reduce, capture rows <= 320) + 0152/0154/0155/0156 (upstream cache fixes)
#   + 0158 (the eagle block-drop keeps the state one block before the dropped block, so a re-sent prefix hits). MTP needs no drafter weights
#   and no /draft mount; VLLM_SM12X_PCIE_IPC_MTP=1 admits it on the pcie_ipc path and both boot lines are asserted below.
#   Why (R206, paired on :8029 the same night, SEQS 16, same pin/knobs): pool 1,309,368 vs 1,052,277 (+24%), code c16 +11%, prose c8 +12%,
#   prose c16 +23%, acceptance per draft token 0.69 vs 0.40 (code); cost: code c1 -24% and +0.26 s on a warm revisit. tool-eval and the
#   fidelity rulers land in the same band (dense corpus PPL 5.3019 vs the R203 pair 5.3062 / 5.3010, inside the daily's own two-boot noise).
#   R206c (SEQS 64 arms): MTP serves c32 (+36% code / +56% prose) and c64 (64 running, 0 preemptions) where DFlash stops admitting at 36.
#   The daily stays at SEQS 16 — the configuration R206/R206b actually measured. SEQS > 40 on this route needs the launcher to cap
#   `max_cudagraph_capture_size` at 320 (the pcie_ipc slab holds 320 rows; 0148 raises ValueError above it) and a pinned KV_BYTES case.
#   Block 1,552 -> 1,472 (the mamba page follows the speculative slot count), so native-l2 is wiped on the first boot, as at R182/R197.
#   The MTP CPU tier holds 595 blocks at 16 GiB (DFlash 1,921): an eagle block carries the speculative state copies, so each tier block is
#   ~3.2x larger. 595 x 1,472 = 875K tokens of staging, still enough to serve 131K-262K prompts (R206 needles 4/4 cold + 4/4 tier-served).
# Rollback: bash /srv/qwen5090/launch-daily-r197-dflash-ns7-0906.sh (DFlash2 ns7 on the R195 image, block 1,552 — the tier is wiped again on
#   the way back; tear this one down first).
set -uo pipefail
GEN_TEMP_IN=${GEN_TEMP:-}   # R777: read BEFORE the unset block below, which runs for EXP boots too (the first R777 V2 t0.3 boot lost it)
EXP=${EXP:-0}; EXP_SEQS=${SEQS:-8}; KV_BYTES=${KV_BYTES:-}; OFFLOAD=${OFFLOAD:-1}; SPLIT_KV_IN=${SPLIT_KV:-}; SSM_DTYPE=${SSM_DTYPE:-bfloat16}; KV_DTYPE=${KV_DTYPE:-nvfp4}; EFFORT=${EFFORT:-medium}  # R285 2026-09-12: EFFORT is read HERE, before the unset block, because R229 introduced the knob and added it to that list in the same commit -- so `${EFFORT:+...}` always expanded to nothing and the knob never once reached the engine. The xhigh SWE-bench campaign failed on this assert on 2026-09-09 and again today; every SWE-bench number on this box was taken at medium.  # R282: KV_DTYPE is EXP-only (the daily branch forces nvfp4 below); AEON needs fp8_e4m3 because nvfp4 KV corrupts that checkpoint (R266). Same not-in-the-unset-list handling as SSM_DTYPE.  # R182; experiments may pass SSM_DTYPE=float32 (NOT in the unset list below: R183 found every EXP boot dying on "SSM_DTYPE: unbound"; the daily branch forces bfloat16)
# Daily: pool band 1.46M–1.525M around 1,490,437 (2026-10-20 pin, MTP ns3, SEQS 16, bf16 SSM) and a 512 MiB free floor. Experiments
# keep an 850K–1.1M band (fp32-SSM pins land at ~904K / ~869K, bf16-SSM at ~986K–1.02M) and the 384 MiB Bug C floor (the r17x boot_cand retry ladders expect it).
DAILY_METHOD=mtp; DAILY_NS=3; DAILY_BLOCK=1472   # R207: MTP head ns3 -> 1,472-token attention block (dflash ns7 gave 1,552, ns9 1,584). Every tier hash carries the block size.
if [ "$EXP" = 0 ]; then MIN_FREE_MIB=${MIN_FREE_MIB:-512}; P_MIN=1460000; P_MAX=1525000; else MIN_FREE_MIB=${MIN_FREE_MIB:-384}; P_MIN=${POOL_MIN:-850000}; P_MAX=${POOL_MAX:-1400000}; fi   # R210: the 1.1M ceiling still rejected an EXP boot of the DAILY shape -- the R197 note below only widened the band for arms with a
# DIFFERENT spec method/ns, and since R207 the EXP default IS the daily (MTP ns3, 1,309,368). POOL_MIN/POOL_MAX now always override on EXP.
# R197: the pool follows the speculative configuration (each slot holds a GDN state snapshot; the MTP head needs no drafter weights): MTP ns3
# booted at 1,309,368 and the 1.1M ceiling rejected a healthy engine. EXP arms with another spec method/length get a wide band and log the pool.
if [ "$EXP" != 0 ] && { [ "${SPEC_METHOD:-$DAILY_METHOD}" != "$DAILY_METHOD" ] || [ "${SPEC_NS:-$DAILY_NS}" != "$DAILY_NS" ]; }; then P_MIN=${POOL_MIN:-600000}; P_MAX=${POOL_MAX:-1500000}; fi
DAILY_IMG=vllm-qwen38:v0310-r27c-A_S-BA4-L1a-ubplan   # BA4 (2026-10-20) = L1a + the BA4 layer; r27 L1a (2026-10-17) was vllm-qwen38:v0310-r27c-A_S-L1a-ubplan = A_S + the L1a draft-plan layer; A_S was vllm-qwen38:v0310-r27c-A_S-splitkv; r27 A_S (2026-10-16): v0.31.0 + patches-v0310 + FlashInfer 0.7.0.post1 + the 0136 lambda fix; the v0.29 R207 image was vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616-pcieipc-bsshash-mtppcie-mtpcache-eagleshift
# Tier knobs (0137): the daily boots with a 300 GB LRU cap on the 393 GB native-l2 fs, evict_scope root (stale namespaces from
# other configs are evicted too), 40 GB min-free (the launcher's own GC threshold). Experiments (EXP≠0) honour the env, unset = no cap.
if [ "$EXP" != 0 ] || [ "${DAILY_ALLOW_ENV:-0}" = 1 ]; then T_CAP=${TIER_CAP_GB:-}; T_SCOPE=${TIER_EVICT_SCOPE:-}; T_MINFREE=${TIER_MIN_FREE_GB:-}; CPU_B=${CPUB:-17179869184}; IMG=${CAND_IMG:-$DAILY_IMG}
else T_CAP=300; T_SCOPE=root; T_MINFREE=40; CPU_B=17179869184; IMG=$DAILY_IMG; fi
# r27 A_S: the v0.31 route keys on the image tag, not on EXP, so an EXP boot of the default image runs the daily's argv and a
# CAND_IMG=<v0.29 image> control keeps the v0.29 argv (its vLLM has no --mamba-backend/--gdn-prefill-backend flags).
case "$IMG" in vllm-qwen38:v0310*) V0310=1;; *) V0310=0;; esac
V0310_ENV=""; V0310_ARGS=""
if [ "$V0310" = 1 ]; then V0310_ENV="-e VLLM_USE_V2_MODEL_RUNNER=1"; V0310_ARGS="--mamba-backend triton --gdn-prefill-backend flashinfer"; fi
# The tier stamp carries the route. The fs layout is NOT what differs: v0.29.0rc2 and v0.31.0 ship the same
# kv_offload/file_mapper.py (config.json, tokens_per_hash), so both routes hash to the same _model_* namespace dir. The
# wipe stays because the block contents one route writes (NVFP4 KV bytes, GDN state snapshots under the V2 runner / triton
# mamba backend / FlashInfer 0.7) have never been validated when read by the other, so switching routes wipes the tier once.
# Daily boots stamp native-l2 (written after the block assert below); EXP boots stamp eval-l2 the same way, written before
# the boot (the route is fixed by the image tag), so alternating CAND_IMG=<v0.29> controls and default A_S boots each start
# that scratch tier cold instead of reading the other route's blocks. Same-route boots never wipe either tier.
DAILY_TIER_STAMP=$DAILY_BLOCK; [ "$V0310" = 0 ] || DAILY_TIER_STAMP=$DAILY_BLOCK-v0310
if [ "${DAILY_ALLOW_ENV:-0}" != 1 ]; then
  unset MODEL_DIR IMAGE KVD_OVERRIDE MAXLEN UTIL NS SPEC_JSON NOSPEC EXTRA_ENV EXTRA_MOUNT EXTRA_ARGS \
        POOL_MIN POOL_MAX TP PP FIWS NO_TIER PIP_ARM CGMODE FUSIONS MNBT SEQS PREFIX_CACHE EAGER MMLIMIT MMKW CPUB \
        MAMBA_MODE GATE_KB PORT NAME BIND_ADDR L2MNT CACHE_DIR ALLOW_NO_XQA ALLOW_NO_PREWARM HEALTH_TRIES \
        TIER_CAP_GB TIER_EVICT_SCOPE TIER_MIN_FREE_GB GEN_TEMP 2>/dev/null || true
fi
# R234 2026-09-09: SEQS 16 pin 13.98 -> 14.86 GB/GPU, pool 1,309,368 -> 1,391,795 (+82,427, +6.3%), band moved with
#   it. The pool follows this pin, not the weights, so NVIDIA's ~880 MiB/GPU of smaller weights (R230: free VRAM 4,413
#   vs RedHat's 3,533) bought free VRAM and no pool until this line changed. 14.86 leaves the daily at 3,581 MiB free
#   after pre-warm and 3,579 MiB through a five-concurrent-120K + one-250K stress -- i.e. at the headroom RedHat
#   already served in production (R232 arm A, results/2026-09-09-r232-nvidia-pool-ladder).
#   WHY NOT HIGHER: not VRAM. R233 (results/2026-09-09-r233-nvidia-pin-clean) booted each higher pin three times with
#   the tier wiped and per-boot seeds: 15.90 GB = 2/3 clean, 16.90 GB = 1/3, both still holding >= 1,371 MiB through
#   the stress. The R191 warmup flake ("CUDA error: invalid argument") rises with the pin, and a 1-in-3 flake makes
#   every daily restart a coin toss. 14.86 is the largest pin with a clean boot record. Rollback: launch-daily-nvidia-1398-0909.sh.
DRAFT=/srv/qwen5090/models/dflash2-qwen38-syvai-w4a16
# R231 2026-09-09 (user call): DAILY CHECKPOINT = nvidia/Qwen3.8-27B-NVFP4 (ModelOpt 0.47, NVFP4 g16 on all 64 MLP
# layers + NVFP4 lm_head + W4A4, FP8 attention; kv_cache_quant_algo null, so the KV dtype stays this launcher's nvfp4).
# Promoted on perf/pool parity and a clean gate sweep, NOT on fidelity: it is +1.83% corpus PPL from bf16 where the
# RedHat checkpoint it replaces is +0.75% (R226b2, results/2026-09-09-r226b2-nvidia; two-boot floor is 0.10-0.15%).
# SWE-bench cannot adjudicate the difference: 387/500 here vs 386-388 for three other checkpoints spanning that whole
# fidelity range (R227). Pool is unchanged at 1,309,368 - it follows --kv-cache-memory-bytes, not the weights; NVIDIA's
# smaller weights show up as free VRAM, and raising the pin is a separate experiment. Rollback: launch-daily-redhat-mtp-0909.sh.
MODEL=/srv/qwen5090/models/qwen3.8-27b-nvidia-nvfp4
# R196 EXP-only passthrough: CAND_MODEL=<dir> audits another checkpoint on the daily route (same KV dtype, drafter, knobs); the daily port ignores it.
if [ "$EXP" != 0 ] && [ -n "${CAND_MODEL:-}" ]; then MODEL=$CAND_MODEL; fi
if [ "$EXP" = 1 ]; then PORT=8029; NAME=vllm-exp; BIND=${EXP_BIND:-127.0.0.1}; L2=/srv/qwen5090/eval-l2; SEQS=$EXP_SEQS
elif [ "$EXP" = eval ]; then PORT=8030; NAME=vllm-eval; BIND=${EVAL_BIND:-127.0.0.1}; L2=/srv/qwen5090/eval-l2; SEQS=$EXP_SEQS
else PORT=8020; NAME=vllm-27b; BIND=0.0.0.0; L2=/srv/qwen5090/native-l2; SEQS=16; SSM_DTYPE=bfloat16; KV_DTYPE=nvfp4; EFFORT=medium; fi   # SEQS 16 since 2026-09-04 (R176, user): R159 c16 = +34% aggregate on the fp8 shape; pin 13.98 GB → pool 903,793 (−3.6% vs SEQS 8)   # BIND 0.0.0.0 deliberate: owui-proxy/harbor reach the engine via 172.17.0.1
ST_=$(cat "$L2/.block" 2>/dev/null || true); if [ "$ST_" != "$DAILY_TIER_STAMP" ]; then
  echo "$(basename "$L2") block stamp '${ST_:-none}' != $DAILY_TIER_STAMP: wiping the tier's _model_* content (was $(du -sh "$L2" 2>/dev/null | cut -f1); R182/R197 — the tier cannot serve blocks of another size; r27 A_S — nor blocks another route wrote)"
  sudo find "$L2" -mindepth 1 -maxdepth 1 -name '_model_*' -exec rm -rf {} + ; sync
  [ "$EXP" = 0 ] || echo "$DAILY_TIER_STAMP" | sudo tee "$L2/.block" >/dev/null; fi   # eval-l2: route stamp only (EXP arms with another ns/TP write their own block size into the tier hash)
if [ -z "$KV_BYTES" ]; then case "$SEQS" in
  8) KV_BYTES=14500000000;; 16) KV_BYTES=15900000000;; 32) KV_BYTES=13440000000;;   # bytes unchanged by R182; the 8/32 pools were read with fp32 SSM (937,795 / ~869K) and are ~+13% with bf16
  *) echo "FAILED: no pinned KV budget for SEQS=$SEQS (set KV_BYTES)"; exit 1;; esac; fi
OFFLOAD_ARGS=""; [ "$OFFLOAD" = 1 ] && OFFLOAD_ARGS="--offload-backend uva --cpu-offload-gb 1 --cpu-offload-params embed_tokens"
case "$SSM_DTYPE" in bfloat16|float32|float16|auto) ;; *) echo "FAILED: SSM_DTYPE must be bfloat16|float32|float16|auto (got $SSM_DTYPE)"; exit 1;; esac
case "$KV_DTYPE" in nvfp4|fp8_e4m3|fp16|auto) ;; *) echo "FAILED: KV_DTYPE must be nvfp4|fp8_e4m3|fp16|auto (got $KV_DTYPE)"; exit 1;; esac
case "$EFFORT" in xhigh|medium|low) ;; *) echo "FAILED: EFFORT must be xhigh|medium|low (got $EFFORT)"; exit 1;; esac
SSM_ARGS="--mamba-ssm-cache-dtype $SSM_DTYPE"
# 0136: FlashInfer 0.6.18 and 0.7 force split-KV OFF for NVFP4 KV in the FA2 prefill wrapper (= the spec-verification path); SPLIT_KV=1
# re-enables it. The v0.29 S image ships FlashInfer 0.6.16.post3 whose split path is on by default, so that route ran SPLIT_KV=0 (r168e:
# closest to bf16 of the three paths; r173c: its decode dumps are identical to rc2 ON's at ctx0 and 30K). The v0.31 route (FlashInfer
# 0.7.0.post1) needs SPLIT_KV=1 to get the same split-KV verify attention back (r27 A_S, 2026-10-16; header). The daily forces the
# route's value; experiments default to it and may pass SPLIT_KV=0|1.
if [ "$EXP" = 0 ]; then SPLIT_KV=$V0310; else SPLIT_KV=${SPLIT_KV_IN:-$V0310}; fi
case "$SPLIT_KV" in 0|1) ;; *) echo "FAILED: SPLIT_KV must be 0 or 1 (got $SPLIT_KV)"; exit 1;; esac
SPLIT_ENV=""; [ "$SPLIT_KV" = 1 ] && SPLIT_ENV="-e VLLM_SM12X_NVFP4_PREFILL_SPLIT_KV=1"
# r27 L1a: the draft-plan layer keys on the image tag; the daily forces it on, EXP boots of that image default to on and may pass
# DRAFT_UB_PLAN=0 (= A_S behaviour on the same image). Other images: off (the env would be ignored, the boot assert forbids it).
case "$IMG" in *-L1a-ubplan*) UB_IMG=1;; *) UB_IMG=0;; esac
if [ "$EXP" = 0 ]; then UB_PLAN=$UB_IMG; else UB_PLAN=${DRAFT_UB_PLAN:-$UB_IMG}; fi
case "$UB_PLAN" in 0|1) ;; *) echo "FAILED: DRAFT_UB_PLAN must be 0 or 1 (got $UB_PLAN)"; exit 1;; esac
if [ "$UB_PLAN" = 1 ] && [ "$UB_IMG" = 0 ]; then echo "FAILED: DRAFT_UB_PLAN=1 needs an *-L1a-ubplan image (got $IMG)"; exit 1; fi
[ "$UB_PLAN" = 1 ] && SPLIT_ENV="$SPLIT_ENV -e VLLM_SM12X_DRAFT_UB_PLAN=1"
# BA4 (2026-10-20): keys on the image tag like L1a; the daily forces it on, EXP boots default on and may pass BA4=0.
case "$IMG" in *-BA4-*) BA4_IMG=1;; *) BA4_IMG=0;; esac
if [ "$EXP" = 0 ]; then BA4_ON=$BA4_IMG; else BA4_ON=${BA4:-$BA4_IMG}; fi
case "$BA4_ON" in 0|1) ;; *) echo "FAILED: BA4 must be 0 or 1 (got $BA4_ON)"; exit 1;; esac
if [ "$BA4_ON" = 1 ] && [ "$BA4_IMG" = 0 ]; then echo "FAILED: BA4=1 needs a *-BA4-* image (got $IMG)"; exit 1; fi
BA4_MOUNT=""
if [ "$BA4_ON" = 1 ]; then
  [ -f /srv/qwen5090/ba4-inputs/binding.json ] || { echo "FAILED: /srv/qwen5090/ba4-inputs missing (copy the screen's preflight/ba4-inputs)"; exit 1; }
  SPLIT_ENV="$SPLIT_ENV -e VLLM_SM12X_BA_M4=1 -e VLLM_SM12X_BA_M4_RESULT=/ba4-results/ba4-rank{rank}.json -e VLLM_SM12X_BA_M4_BINDING=/ba4-results/binding.json"
  BA4_MOUNT="-v /srv/qwen5090/ba4-inputs:/ba4-results:ro"
fi
# R185: the daily forces the pcie_ipc all-reduce on; experiments opt in with PCIE_IPC=1 (an image without 0138 then fails the assert below).
# (R187 fix 2026-09-05: the first version assigned 1 and then read its own value, so every experiment saw PCIE_IPC=1 and images without 0138 failed the assert)
if [ "$EXP" = 0 ]; then PCIE_IPC=1; else PCIE_IPC=${PCIE_IPC:-0}; fi
case "$PCIE_IPC" in 0|1) ;; *) echo "FAILED: PCIE_IPC must be 0 or 1 (got $PCIE_IPC)"; exit 1;; esac
PCIE_ENV=""; [ "$PCIE_IPC" = 1 ] && PCIE_ENV="-e VLLM_SM12X_PCIE_IPC_AR=1"
XARGS=""; XMOUNT=""; MNBT_=8192; FUS_=""; SPX_=""; XENV=""; CCX_=""
# R183 EXP-only passthrough (the EXP=0 path is unchanged): EXP_MNBT (chunk size; the daily's 8192 assert follows it), FUSIONS_APPEND (raw
# pass_config pairs -> v0280 FUSIONS), SPEC_EXTRA (raw JSON pairs appended inside --speculative-config), EXTRA_ENV_APPEND (-e pairs), CC_EXTRA (raw top-level
# compilation-config pairs -> v0280 CCEXTRA).
if [ "$EXP" != 0 ]; then XARGS="${EXTRA_ARGS_APPEND:-}"; XMOUNT="${EXTRA_MOUNT_APPEND:-}"; MNBT_=${EXP_MNBT:-8192}; FUS_="${FUSIONS_APPEND:-}"; SPX_="${SPEC_EXTRA:-}"; XENV="${EXTRA_ENV_APPEND:-}"; CCX_="${CC_EXTRA:-}"; fi
# 2026-10-20 (no block drop): the MTP speculative config carries "disable_eagle_block_drop":true. Without it vLLM drops the last
#   full-attention block of every prefix hit and re-prefills it, and on 32K revisits the greedy continuation differed from the
#   cold one in 11 of 12 prompts; with it 12/12 identical, warm TTFT 0.275 -> 0.071 s (32K) and 0.265 -> 0.121 s (6K), decode
#   unchanged (results 2026-10-20-051649-r27-flags-screen, 2026-10-20-070708-r27-flags-screen, 2026-10-20-023934-r27-flags-screen;
#   bench/results/r27-no-block-drop.md). The daily forces it; EXP boots default to it too (NO_BLOCK_DROP=0 gives the old drop for
#   A/B arms).
if [ "$EXP" = 0 ]; then NBD_=1; else NBD_=${NO_BLOCK_DROP:-1}; fi
if [ "$NBD_" = 1 ]; then case "$SPX_" in *disable_eagle_block_drop*) ;; *) SPX_="${SPX_:+$SPX_,}\"disable_eagle_block_drop\":true";; esac; fi
case "$MNBT_" in *[!0-9]*|"") echo "FAILED: EXP_MNBT must be an integer (got $MNBT_)"; exit 1;; esac
# R195: the daily forces batch-sharded sampling on; experiments opt in with BSS=1 or by passing the flag in EXTRA_ARGS_APPEND (r191/r193* do).
if [ "$EXP" = 0 ]; then BSS=1; else BSS=${BSS:-0}; case "${XARGS:-}" in *enable-batch-sharded-sampling*) BSS=1;; esac; fi
case "$BSS" in 0|1) ;; *) echo "FAILED: BSS must be 0 or 1 (got $BSS)"; exit 1;; esac
BSS_ARGS=""; if [ "$BSS" = 1 ]; then case "${XARGS:-}" in *enable-batch-sharded-sampling*) ;; *) BSS_ARGS="--enable-batch-sharded-sampling";; esac; fi
NS_=$DAILY_NS; DTP_=2; SPEC_METHOD_=$DAILY_METHOD; if [ "$EXP" != 0 ]; then NS_=${SPEC_NS:-$DAILY_NS}; DTP_=${SPEC_DTP:-2}; SPEC_METHOD_=${SPEC_METHOD:-$DAILY_METHOD}; fi   # R207: EXP defaults follow the daily; a DFlash control must pass SPEC_METHOD=dflash SPEC_NS=7 (and CAND_IMG=...-pcieipc-bsshash for the pre-R207 image — dflash on the 0158 image is UNMEASURED)
# R210 EXP-only passthrough: EXP_TP and EXP_MAXLEN. They are EXP_-prefixed because this block runs AFTER the unset list
# above, which wipes TP, POOL_MIN and POOL_MAX on the experiment path too (it exists so no knob leaks into a daily
# restore) -- reading $TP here silently yielded 2 and R210b booted a whole TP=2 ladder believing it was measuring one card. The daily is TP=2 at 262,144 by contract and both are unset on its
# branch above, so this can only loosen the experiment path. TP=1 also forces the drafter to one card and forbids the
# pcie_ipc all-reduce (a two-card kernel), and its pool is ~5x smaller, so the pool band has to open too.
TP_=2; MAXLEN_=262144
if [ "$EXP" != 0 ]; then
  TP_=${EXP_TP:-2}; MAXLEN_=${EXP_MAXLEN:-262144}
  if [ "$TP_" != 2 ]; then
    [ "$PCIE_IPC" = 0 ] || { echo "FAILED: pcie_ipc is a two-card all-reduce; pass PCIE_IPC=0 with TP=$TP_"; exit 1; }
    [ "$BSS" = 0 ] || { echo "FAILED: batch-sharded sampling has nothing to shard at TP=$TP_ (vLLM rejects the flag outright); pass BSS=0"; exit 1; }
    DTP_=${SPEC_DTP:-$TP_}; P_MIN=${EXP_POOL_MIN:-100000}; P_MAX=${EXP_POOL_MAX:-1500000}
  fi
fi
# R197 EXP-only passthrough: SPEC_METHOD=mtp swaps the DFlash2 drafter for the checkpoint's own MTP head (vLLM method qwen3_5_mtp, SPEC_NS
# tokens, no /draft mount; the drafter-graph asserts below apply to dflash only). The daily port runs MTP ns3 since R207 (dflash ns7 until then).
# R203 EXP-only passthrough: SPEC_METHOD=none boots with no speculative decoding at all (NOSPEC=1 in launch-daily-v0280.sh, the R115 A-arm switch):
# no drafter mount, no --speculative-config, no drafter graphs; the attention block follows the spec-free mamba page (value logged, not asserted).
case "$SPEC_METHOD_" in dflash|mtp|none) ;; *) echo "FAILED: SPEC_METHOD must be dflash, mtp or none (got $SPEC_METHOD_)"; exit 1;; esac
[ "$SPEC_METHOD_" != none ] || [ "$EXP" != 0 ] || { echo "FAILED: SPEC_METHOD=none is EXP-only"; exit 1; }
NOSPEC_=0
case "$NS_$DTP_" in *[!0-9]*) echo "FAILED: SPEC_NS/SPEC_DTP must be integers (got $NS_/$DTP_)"; exit 1;; esac
if [ "$SPEC_METHOD_" = none ]; then
  SPEC_JSON_=""; DRAFT_MOUNT=""; SPEC_DESC="no speculative decoding"; NOSPEC_=1
elif [ "$SPEC_METHOD_" = mtp ]; then
  SPEC_JSON_='{"method":"qwen3_5_mtp","num_speculative_tokens":'$NS_"${SPX_:+,$SPX_}"'}'; DRAFT_MOUNT=""; SPEC_DESC="MTP head ns$NS_$([ "$NBD_" = 1 ] && echo ' no-block-drop')"
else
  SPEC_JSON_='{"method":"dflash","model":"/draft","num_speculative_tokens":'$NS_',"draft_tensor_parallel_size":'$DTP_',"attention_backend":"FLASHINFER"'"${SPX_:+,$SPX_}"'}'; DRAFT_MOUNT="-v $DRAFT:/draft:ro"; SPEC_DESC="DFlash2 ns$NS_ draft_tp$DTP_ in CUDA graphs"
fi
# R207/0148: the sequential MTP drafter is only admitted on the pcie_ipc all-reduce behind its own switch (without it the drafter falls back
# to the CUSTOM/PYNCCL path and the "MTP drafter admitted" line below is absent).
if [ "$PCIE_IPC" = 1 ] && [ "$SPEC_METHOD_" = mtp ]; then PCIE_ENV="$PCIE_ENV -e VLLM_SM12X_PCIE_IPC_MTP=1"; fi
[ -f "$MODEL/model.safetensors.index.json" ] || [ -f "$MODEL/model.safetensors" ] || { echo "FAILED: checkpoint missing at $MODEL"; exit 1; }
# Every boot compiles at STOCK power limits, whatever the box was left at. torch.compile autotunes against the
# prefill-shaped pre-warm forward and the artifact is already a timing lottery (R193), so the power state during a boot
# must not depend on whether an experiment or a capped daily ran before it. The daily gets its 400 W cap further down,
# after its asserts pass; experiments keep stock.
bash /srv/qwen5090/daily-power.sh stock | grep -i '^WARN' || true
# Clock offsets are per card and shared with the Flash-Next daily (:8022), so this launcher sets its own at every boot.
# User 2026-09-25: memory +4500 on both dailies, core overclock ONLY on Flash-Next -> the 27B boots at core offset 0.
# (R726: the +4500 applied once by gpu-tune.service had silently reset to 0 between 09-03 and 09-19.)
CLK=$(sudo -n python3 -c "import pynvml as N; N.nvmlInit(); hs=[N.nvmlDeviceGetHandleByIndex(i) for i in range(N.nvmlDeviceGetCount())]; [(N.nvmlDeviceSetMemClkVfOffset(h, 4500), N.nvmlDeviceSetGpcClkVfOffset(h, 0)) for h in hs]; print('mem', *[N.nvmlDeviceGetMemClkVfOffset(h) for h in hs], '| core', *[N.nvmlDeviceGetGpcClkVfOffset(h) for h in hs])" 2>&1 | tail -1)
echo "clock offsets: memory +4500, core 0 on every GPU; readback $CLK"
sudo docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FAILED: image $IMG missing (build: v0.31 = scripts/build-v031-image.sh BA4=1, patches-v0310/README.md; v0.29 = build-v0290rc2.sh + the fi0616 swap layer + patches-v0290/Dockerfile.pcieipc + Dockerfile.bss-not-a-compile-factor)"; exit 1; }
GT_=""; [ "$EXP" != 0 ] && GT_=$GEN_TEMP_IN   # R777 2026-09-27: EXP-only default temperature (the daily keeps v0280's 0.6)
env PORT=$PORT NAME=$NAME BIND_ADDR=$BIND MODEL_DIR="$MODEL" TP=$TP_ L2MNT="$L2" CPUB=$CPU_B EFFORT=$EFFORT ${GT_:+GEN_TEMP=$GT_} ${T_CAP:+TIER_CAP_GB=$T_CAP} ${T_SCOPE:+TIER_EVICT_SCOPE=$T_SCOPE} ${T_MINFREE:+TIER_MIN_FREE_GB=$T_MINFREE} \
    IMAGE="$IMG" KVD_OVERRIDE=$KV_DTYPE ALLOW_NO_XQA=1 \
    NO_TIER=0 FIWS=536870912 MNBT=$MNBT_ SEQS=$SEQS ${FUS_:+FUSIONS="$FUS_"} ${CCX_:+CCEXTRA="$CCX_"} UTIL=0.88 MAXLEN=$MAXLEN_ POOL_MIN=$P_MIN POOL_MAX=$P_MAX \
    EXTRA_MOUNT="$DRAFT_MOUNT $BA4_MOUNT $XMOUNT" \
    EXTRA_ARGS="--kv-cache-memory-bytes $KV_BYTES $OFFLOAD_ARGS $SSM_ARGS $BSS_ARGS $V0310_ARGS $XARGS" \
    SPEC_JSON="$SPEC_JSON_" NOSPEC=$NOSPEC_ \
    EXTRA_ENV="-e NCCL_P2P_LEVEL=SYS -e VLLM_SM12X_NVFP4_XQA=0 -e VLLM_SM12X_DFLASH_GRAPHS=1 $SPLIT_ENV $PCIE_ENV $V0310_ENV $XENV" \
    bash /srv/qwen5090/launch-daily-v0280.sh || { echo "0.29 nvfp4 DAILY FAILED — engine NOT up$([ "$EXP" != 0 ] || echo '; rollback: launch-daily-v0290-r207.sh')"; exit 1; }   # label text kept: r793/r794/r796 grep it
BOOTLOG=$(sudo docker logs "$NAME" 2>&1)
ARGS=$(sudo docker inspect "$NAME" --format '{{json .Args}} {{json .Config.Env}}')
fail(){ echo "FAILED: $1"; exit 1; }
VER=$(sudo docker exec "$NAME" python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null)
FIVER=$(sudo docker exec "$NAME" python3 -c 'import flashinfer; print(flashinfer.__version__)' 2>/dev/null)
# r27 A_S (2026-10-16): the daily image is v0.31.0 with FlashInfer 0.7.0.post1 (the pin the harness asserted); an EXP candidate may
# be 0.29.x (the rollback route), 0.30.x (patches/vllm-v0300) or 0.31.x. A v0310-tagged image must really be 0.31: the V0310
# flags above exist only there.
if [ "$IMG" = "$DAILY_IMG" ]; then case "$VER" in 0.31.0*) ;; *) fail "engine is vllm '$VER', not 0.31.0 (image drift)";; esac
else case "$VER" in 0.29*|0.30*|0.31*) ;; *) fail "candidate engine is vllm '$VER', not 0.29.x/0.30.x/0.31.x";; esac; fi
[ "$V0310" = 0 ] || case "$VER" in 0.31*) ;; *) fail "image tag $IMG selects the v0.31 route but the engine is vllm '$VER'";; esac
if [ "$IMG" = "$DAILY_IMG" ]; then [ "$FIVER" = 0.7.0.post1 ] || fail "A_S image should carry FlashInfer 0.7.0.post1, got '$FIVER' (image drift)"; fi
# fail-closed asserts (grep -c, not -q: pipefail + -q SIGPIPE gotcha)
[ "$(echo "$BOOTLOG" | grep -ac "as specified by kv_cache_memory_bytes")" -ge 1 ] || fail "pinned KV budget not honoured (no kv_cache_memory_bytes line)"
# 0131 shrinks the graph-bound POOLED PREFILL wrappers, which only the spec-verification decode path creates (q = ns+1 per request); a spec-free
# boot (SPEC_METHOD=none, R203) decodes through the FlashInfer decode wrapper and never logs the line — the assert is spec-ON only.
if [ "$SPEC_METHOD_" != none ]; then [ "$(echo "$BOOTLOG" | grep -ac "int workspace shrunk 8 MiB -> 1 MiB")" -ge 1 ] || fail "0131 pooled int workspace not active (image/env drift)"; fi
if [ "$SPEC_METHOD_" = dflash ]; then
  [ "$(echo "$BOOTLOG" | grep -ac "Capturing dflash2 CUDA graphs")" -ge 1 ] || fail "drafter graphs not captured (0129 inactive?)"
  [ "$(echo "$BOOTLOG" | grep -ac "running the draft eagerly")" -eq 0 ] || fail "drafter fell back to eager"
else
  [ "$(echo "$BOOTLOG" | grep -ac "Capturing dflash2 CUDA graphs")" -eq 0 ] || fail "SPEC_METHOD=$SPEC_METHOD_ but the DFlash2 drafter captured graphs"
fi
[ "$(echo "$BOOTLOG" | grep -ac "decode_backend=xqa")" -eq 0 ] || fail "XQA decode engaged — Bug B dodge not in force"
[ "$(echo "$BOOTLOG" | grep -ac "$(basename "$MODEL")\|compressed-tensors\|quantization=modelopt")" -ge 1 ] || fail "checkpoint identity"   # R199: ModelOpt candidates (CAND_MODEL) log quantization=modelopt, never their dir name
[ "$(echo "$ARGS" | grep -ac -- "--max-num-batched-tokens $MNBT_")" -ge 1 ] || fail "MNBT is not $MNBT_ (8192 = 5 blocks on the daily; R211 — see header)"
[ "$(echo "$ARGS" | grep -ac -- "--mamba-ssm-cache-dtype $SSM_DTYPE")" -ge 1 ] || fail "SSM cache dtype is not $SSM_DTYPE on the container"
# R197: the attention block is sized to the mamba page, which grows with the number of speculative slots (ns9 → 1,584; ns7 → 1,552; MTP ns3 → 1,472),
# so the exact-value assert holds for the daily's own method+ns only; EXP arms with another ns/method must still show the sizing line, and log its value.
# R210: TP is the same kind of input -- one rank holds the whole GDN state at TP=1, so the page and the block sized to it differ from the daily's.
if [ "$SSM_DTYPE" = bfloat16 ]; then
  if [ "$EXP" = 0 ] || { [ "$TP_" = 2 ] && [ "$SPEC_METHOD_" = "$DAILY_METHOD" ] && [ "$NS_" = "$DAILY_NS" ]; }; then
    [ "$(echo "$BOOTLOG" | grep -ac "Setting attention block size to $DAILY_BLOCK tokens")" -ge 1 ] || fail "bf16 SSM state at $DAILY_METHOD ns$DAILY_NS should give a $DAILY_BLOCK-token attention block (R180/R197/R207); block line missing or different"
    [ "$EXP" != 0 ] || echo "$DAILY_TIER_STAMP" | sudo tee "$L2/.block" >/dev/null
  else
    BLK_=$(echo "$BOOTLOG" | grep -aoE 'Setting attention block size to [0-9]+ tokens' | head -1 | tr -dc 0-9)
    [ -n "$BLK_" ] || fail "attention block sizing line missing"; echo "EXP $SPEC_DESC: attention block $BLK_ tokens (daily $DAILY_METHOD ns$DAILY_NS = $DAILY_BLOCK)"
  fi
fi
if [ "$SPEC_METHOD_" = dflash ]; then
  [ "$(echo "$ARGS" | grep -acE "num_speculative_tokens.{1,5}$NS_[,}]")" -ge 1 ] && [ "$(echo "$ARGS" | grep -acE "draft_tensor_parallel_size.{1,5}$DTP_[,}]")" -ge 1 ] || fail "speculative config is not ns$NS_ draft_tp$DTP_ on the container"
elif [ "$SPEC_METHOD_" = mtp ]; then
  [ "$(echo "$ARGS" | grep -acE "num_speculative_tokens.{1,5}$NS_[,}]")" -ge 1 ] && [ "$(echo "$ARGS" | grep -ac "qwen3_5_mtp")" -ge 1 ] || fail "speculative config is not MTP ns$NS_ on the container"
  [ "$NBD_" != 1 ] || [ "$(echo "$ARGS" | grep -acE 'disable_eagle_block_drop.{1,5}true')" -ge 1 ] || fail "disable_eagle_block_drop is not on the container"
else
  [ "$(echo "$ARGS" | grep -ac "speculative-config")" -eq 0 ] || fail "SPEC_METHOD=none but a --speculative-config is on the container"
  [ "$(echo "$ARGS" | grep -ac "num_speculative_tokens")" -eq 0 ] || fail "SPEC_METHOD=none but num_speculative_tokens is on the container"
fi
[ "$(echo "$ARGS" | grep -ac "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=536870912")" -ge 1 ] || fail "FlashInfer workspace is not 512 MiB (Bug B dodge)"
[ "$(echo "$ARGS" | grep -ac "VLLM_SM12X_NVFP4_XQA=0")" -ge 1 ] || fail "VLLM_SM12X_NVFP4_XQA=0 missing"
# CPU tier (r172): every disk-tier hit is promoted through the CPU tier, so 16 GiB is what lets 131K–262K prompts be served. The block count
# depends on the route: dflash stores ~1,010 blocks of 2,944 / ~1,880 of 1,584 / ~1,921 of 1,552, while an eagle/MTP block also carries the
# speculative state copies and is ~3.2x larger (R207: 595 blocks of 1,472 = 875K tokens, still 4/4 tier-served needles at 131K/220K in R206).
# Measured on disk: an MTP tier block file is 28,827,648 B for 1,472 tokens (19.6 KB/token) vs dflash 8.9 MB for 1,552 (5.8 KB/token), so the
# 300 GB disk cap holds ~15M tokens instead of ~52M. Read volume per served token did NOT rise (2,529 MB / 129,536 tok vs 2,968 MB / 130,368).
# v0.31 logs the same count as "chunks" ("primary tier (lru, 595 chunks)", r27 audition 2026-10-13); both spellings are accepted.
[ "$(echo "$ARGS" | grep -acE "cpu_bytes_to_use.{1,5}$CPU_B[,}]")" -ge 1 ] || fail "CPU tier is not $CPU_B bytes on the container"
CPUBLK=$(echo "$BOOTLOG" | grep -aoE 'primary tier \(lru, [0-9]+ (blocks|chunks)\)' | tail -1 | grep -oE '[0-9]+')
MIN_CPUBLK=900; [ "$SPEC_METHOD_" != mtp ] || MIN_CPUBLK=550
[ "$EXP" != 0 ] || [ "${CPUBLK:-0}" -ge "$MIN_CPUBLK" ] || fail "CPU tier has ${CPUBLK:-?} blocks (expected >= $MIN_CPUBLK for $SPEC_METHOD_: ~1,921 of 1,552 on dflash, 595 of 1,472 on MTP)"
if [ -n "$T_CAP" ]; then [ "$(echo "$ARGS" | grep -acE "max_capacity_gb.{1,5}$T_CAP[,}]")" -ge 1 ] || fail "0137 tier cap $T_CAP GB not on the container"; fi
if [ "$OFFLOAD" = 1 ]; then  # R167 / NOTES18 §5, fail closed
  [ "$(echo "$BOOTLOG" | grep -ac 'Offloader set to UVAOffloader')" -ge 1 ] || fail "UVAOffloader not selected (0135 inactive?)"
  if [ "$TP_" = 2 ]; then   # 1.18 GiB is the PER-RANK half of the embedding; at TP=1 one rank holds all of it (R210)
    [ "$(echo "$BOOTLOG" | grep -ac 'Total CPU offloaded parameters: 1.18')" -ge 1 ] || fail "target embed shard not offloaded (no 'Total CPU offloaded parameters: 1.18')"
  else
    OFFG_=$(echo "$BOOTLOG" | grep -aoE 'Total CPU offloaded parameters: [0-9.]+' | tail -1 | grep -oE '[0-9.]+$')
    [ -n "$OFFG_" ] || fail "no 'Total CPU offloaded parameters' line at TP=$TP_"
    echo "EXP TP=$TP_: CPU offloaded parameters $OFFG_ GiB (TP=2 shards the embedding to 1.18 per rank)"
  fi
  [ "$(echo "$BOOTLOG" | grep -ac 'matched no parameters')" -eq 0 ] || fail "offload selector matched no parameters"
  [ "$(echo "$ARGS" | grep -ac 'VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1\|VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1')" -eq 0 ] || fail "UVA/pinning disabled by env"
else
  [ "$(echo "$BOOTLOG" | grep -ac 'CPU offloaded parameters')" -eq 0 ] || fail "OFFLOAD=0 but the engine offloaded parameters"
fi
if [ "$SPLIT_KV" = 1 ]; then [ "$(echo "$BOOTLOG" | grep -ac "re-enabled FlashInfer split-KV")" -ge 1 ] || fail "SPLIT_KV=1 but 0136 did not engage (image without 0136?)"
else [ "$(echo "$BOOTLOG" | grep -ac "re-enabled FlashInfer split-KV")" -eq 0 ] || fail "split-KV re-enabled although SPLIT_KV=0"; fi
if [ "$UB_PLAN" = 1 ]; then  # r27 L1a, fail closed: both lines once per worker (lever-arms.json admission_log), never INACTIVE
  [ "$(echo "$BOOTLOG" | grep -ac "SM12X draft UB plan ACTIVE: VLLM_SM12X_DRAFT_UB_PLAN=1")" -ge 1 ] || fail "DRAFT_UB_PLAN=1 but no 'SM12X draft UB plan ACTIVE' line (image without the L1a layer?)"
  [ "$(echo "$BOOTLOG" | grep -ac "SM12X draft UB plan USED")" -ge 1 ] || fail "DRAFT_UB_PLAN=1 but the draft-decode plan never used the upper bound (no 'SM12X draft UB plan USED' line)"
  [ "$(echo "$BOOTLOG" | grep -ac "SM12X draft UB plan INACTIVE")" -eq 0 ] || fail "SM12X draft UB plan INACTIVE logged"
else [ "$(echo "$BOOTLOG" | grep -ac "SM12X draft UB plan ACTIVE")" -eq 0 ] || fail "draft UB plan engaged although DRAFT_UB_PLAN=0"; fi
if [ "$PCIE_IPC" = 1 ]; then  # R185, fail closed: a silent fallback to CustomAllreduce is the failure mode this guards
  [ "$(echo "$BOOTLOG" | grep -ac "PCIe IPC all-reduce enabled")" -ge 1 ] || fail "PCIE_IPC=1 but no 'PCIe IPC all-reduce enabled' line (image without 0138, or the kernel fell back to CUSTOM)"
  [ "$(echo "$BOOTLOG" | grep -acF "Using ['PCIE_IPC', 'CUSTOM', 'PYNCCL'] all-reduce backends")" -ge 1 ] || fail "all-reduce backend order is not PCIE_IPC, CUSTOM, PYNCCL"
  [ "$(echo "$ARGS" | grep -ac "VLLM_SM12X_PCIE_IPC_AR=1")" -ge 1 ] || fail "VLLM_SM12X_PCIE_IPC_AR=1 missing on the container"
  if [ "$SPEC_METHOD_" = mtp ]; then  # R207/0148, fail closed: without the switch the MTP drafter silently leaves the pcie_ipc path
    [ "$(echo "$BOOTLOG" | grep -ac "SM12X PCIe IPC: MTP drafter admitted")" -ge 1 ] || fail "MTP on pcie_ipc: no 'MTP drafter admitted' line (image without 0148, or capture rows > 320 — cap max_cudagraph_capture_size at 320 above SEQS 40)"
    [ "$(echo "$ARGS" | grep -ac "VLLM_SM12X_PCIE_IPC_MTP=1")" -ge 1 ] || fail "VLLM_SM12X_PCIE_IPC_MTP=1 missing on the container"
  fi
else [ "$(echo "$BOOTLOG" | grep -ac "PCIe IPC all-reduce enabled")" -eq 0 ] || fail "pcie_ipc all-reduce engaged although PCIE_IPC=0"; fi
if [ "$BSS" = 1 ]; then  # R195, fail closed (-ge 1: rank 1 does not always log INFO, R190e)
  [ "$(echo "$BOOTLOG" | grep -ac "Batch-sharded sampling enabled")" -ge 1 ] || fail "BSS=1 but no 'Batch-sharded sampling enabled' line"
  [ "$(echo "$ARGS" | grep -ac -- "--enable-batch-sharded-sampling")" -ge 1 ] || fail "--enable-batch-sharded-sampling missing on the container"
else [ "$(echo "$BOOTLOG" | grep -ac "Batch-sharded sampling enabled")" -eq 0 ] || fail "sharded sampling engaged although BSS=0"; fi
# Image identity. v0.31 (patches-v0310 apply_layer.py writes one marker per applied patch): 0136 = split-KV override, 0147 = sharded
# sampling not a compile factor, 0148 = MTP on pcie_ipc, 0154 = Mamba groups not EAGLE; 0152/0155/0156/0158 are upstream in v0.31 and
# the build verified their source into /opt/prs-markers/upstream. The A_S layer's lambda takes FlashInfer 0.7's (kv_data_type, device).
if [ "$IMG" = "$DAILY_IMG" ]; then
  for M_ in 0136 0147 0148 0154; do sudo docker exec "$NAME" test -f /opt/prs-markers/$M_ || fail "daily image lacks the $M_ marker (image drift)"; done
  [ "$(sudo docker exec "$NAME" cat /opt/prs-markers/upstream 2>/dev/null | grep -c '0158')" -ge 1 ] || fail "daily image lacks the upstream-source marker for 0152/0155/0156/0158 (image drift)"
  [ "$(sudo docker exec "$NAME" grep -c 'lambda \*_args, \*\*_kwargs: False' /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flashinfer.py 2>/dev/null)" = 1 ] || fail "A_S split-KV lambda layer missing (FlashInfer 0.7 would raise TypeError at the first plan())"
fi
# R207/0158, fail closed on the v0.29 route: without the replay boundary an eagle/MTP boot drops the block that a re-sent prefix needs and
# every revisit re-prefills (R205: prefix-cache hits 0). The line names the retention interval and the scheduler block size. v0.31 carries
# the boundary upstream (kv_cache_coordinator.get_replay_boundaries) and logs no such line; the r27 QUALIFY warm-equal and boundary gates
# measured its prefix hits instead.
if [ "$SPEC_METHOD_" = mtp ] && [ "$V0310" = 0 ] && [ "$IMG" = vllm-qwen38:v0290rc2-nvfp4kv-revival-prs-fi0616-pcieipc-bsshash-mtppcie-mtpcache-eagleshift ]; then
  [ "$(echo "$BOOTLOG" | grep -ac "SM12X eagle-drop replay boundary retained")" -ge 1 ] || fail "0158 replay boundary not active (MTP prefix hits would be 0)"
fi
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
[ "$FREE" -ge "$MIN_FREE_MIB" ] || fail "only $FREE MiB free after pre-warm (< $MIN_FREE_MIB) — Bug C headroom missing; lower KV_BYTES"
POOL=$(echo "$BOOTLOG" | grep -a 'GPU KV cache size' | tail -1 | grep -oE 'cache size: [0-9,]+' | tr -dc 0-9)
# Power policy (user 2026-09-08): the DAILY runs both cards capped at 400 W, EXPERIMENTS run at stock limits so every
# published number stays comparable with the ones already measured at stock. See daily-power.sh for the R208/R209
# evidence and for why there is no "cap while the container is up" mode. Two steps: stock is set BEFORE the boot above
# (so every boot compiles under identical power, whatever ran before), and the daily is capped HERE, only once its
# asserts have passed. It never fails the boot -- it is a comfort setting, not a gate; daily-power.sh warns and returns 0.
if [ "$EXP" = 0 ]; then bash /srv/qwen5090/daily-power.sh cap | grep -i '^WARN' || true; fi
PLIM=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | tr -d ' ' | paste -sd/ -)
# BA4 never fails the boot: an inactive layer runs the original GEMM (correct, ~12 % slower at c1). Say how many are active.
BA4_DESC=""; if [ "$BA4_ON" = 1 ]; then
  BA4_A=$(echo "$BOOTLOG" | grep -aoE "SM12X BA4 ACTIVE: prefix=[^ ]+ rank=[0-9]" | sort -u | wc -l)
  BA4_I=$(echo "$BOOTLOG" | grep -ac "SM12X BA4 INACTIVE" || true)
  BA4_DESC=" + BA4 $BA4_A/96 active"; [ "$BA4_A" = 96 ] && [ "$BA4_I" = 0 ] || echo "WARN: BA4 $BA4_A/96 rank-layers active, $BA4_I INACTIVE lines (falls back to the original GEMM)"
fi
LABEL="0.29 nvfp4 DAILY UP"; [ "$EXP" = 0 ] || LABEL="0.29 nvfp4 EXP UP"   # literal kept on the v0.31 route: r793/r794/r796 grep it; the line names vllm $VER
echo "$LABEL on ${BIND}:${PORT} (vllm $VER, FlashInfer $FIVER, NVIDIA NVFP4 weights + NVFP4 KV pinned $KV_BYTES B/GPU + $SPEC_DESC + SSM $SSM_DTYPE + 0131/0134 + embed offload=$OFFLOAD + split_kv=$SPLIT_KV + draft_ub_plan=$UB_PLAN + pcie_ipc=$PCIE_IPC + bss=$BSS + native disk tier${T_CAP:+ cap ${T_CAP} GB} + CPU tier $CPU_B B, dual 5090, image $IMG, SEQS $SEQS). Pool $POOL, min free VRAM $FREE MiB, power $PLIM W.$([ "$EXP" != 0 ] || echo ' Rollback: launch-daily-v0290-r207.sh')$BA4_DESC"
