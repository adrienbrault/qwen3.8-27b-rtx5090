# R160, SWE-Bench Verified with mini-SWE-agent on the fp8 daily: 386/500 = 77.2% (2026-09-02→03, `results/2026-09-02-miniswe-rh`, `scripts/miniswe-full.sh`)

[← all results](../RESULTS.md)

Leaderboard-shaped run: mini-swe-agent 2.4.6 (its builtin `swebench.yaml` plus the local model config in `scripts/miniswe/`), litellm 1.99.0, the official swebench 4.1.0 harness in the official `sweb.eval` images, the full Verified 500, one attempt each. Engine: the daily stack on :8030 (RedHatAI NVFP4 weights, fp8 KV, DFlash2 ns9 TP=2, util 0.90, SEQS 16, disk tier on), 16 then 12 agent workers, chunks of 40 scored as they finished (`scripts/miniswe-full.sh`, `scripts/miniswe-score.sh`).

| | |
|---|---|
| resolved | **386 / 500 = 77.2%** |
| completed / empty patches / scoring errors | 498 / 2 / 0 |
| exits | 499 Submitted, 1 RepeatedFormatError |
| per repo | django 179/231 (77.5%), sympy 59/75 (78.7%), sphinx 34/44 (77.3%), matplotlib 23/34 (67.6%), scikit-learn 27/32 (84.4%), astropy 16/22, pydata 16/22 (72.7%), pytest 18/19 (94.7%), pylint 5/10, psf 7/8, mwaskom 1/2, pallets 1/1 |
| pace | ≈2.2 instances/min at 12 workers (a chunk of 40 in 17–42 min), ≈5.5 h of agent time for the 500 |
| engine | 0 runtime error lines; 35 preemptions on the last engine; the disk tier filled to 100% twice and was cycled (wipe + reboot, 4.5 min each) |

Reference points for the number: the same scaffold on Qwen3.6-27B-FP8 scored 67.8% (QwenLM discussion 1846); public SOTA on the leaderboard is 79.2%, engineered multi-attempt stacks report 88–90. This rig's earlier R2E-scaffold runs (66.2% on the saka checkpoint, above) used a different scaffold and are not comparable one-to-one. The run survived an OOM-panic reboot (earlyoom installed since), two heavy-TP2 boot failures at a 45 s settle (see the settle ladder), and a user pause; 23 mid-run scoring errors (Docker Hub 500s on image pulls) cleared on re-score. This is the fp8 half of a pair: the nvfp4-KV candidate runs the identical 500 next and the comparison is per instance.
