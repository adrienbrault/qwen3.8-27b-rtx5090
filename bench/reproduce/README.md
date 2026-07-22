# Reproduce the agentic benchmark scores

Raw artifacts and pinned commands behind the headline numbers, so the 69.4% / 48.3% can be audited rather than trusted. The serving side is always the tier daily from [`../../scripts/serve.sh`](../../scripts/serve.sh) (or a port-shifted copy), model snapshot `natfii/Qwen3.6-27B-VLM-NVFP4-MTP@2e46c0ed7606f35e357bc5674d20c710fc51b178`.

## What's in this directory

| file | what it is |
|---|---|
| `preds.jsonl.gz` | SWE-Bench-Verified predictions, all 500 instances, **as exported by the agent** (pre-sanitization) — `instance_id`, `model_name_or_path`, `model_patch` |
| `preds_fix.jsonl.gz` | The sanitized patches for the 58 instances re-scored after the build-file-hunk strip |
| `qwen3.6-27b-tiers.deepswe.json` | Official `swebench` harness report, first full pass (reads 62.2% — includes the mechanically-zeroed instances) |
| `qwen3.6-27b-tiers.deepswefix.json` | Official harness report for the sanitized re-score → combined final **347/500 = 69.4%** |
| `deepswe-sanitize-rescore.sh` | **The sanitizer** — strips root-level `{tox.ini, pyproject.toml, setup.cfg, setup.py}` hunks from patches that also touch source files (uniformly, all 500), then re-runs the harness on affected-unresolved instances |
| `final-c2-outcomes.txt` | Terminal-Bench 2.1, per-task outcome for all 89 tasks (PASS / FAIL / TIMEOUT / ERR) from the scored c2 run |

Full trajectories (~GBs: per-step agent transcripts, terminal recordings, harbor job dirs, llama-benchy raw logs) don't fit a git repo — open an issue if you want a drop of any specific run.

## SWE-Bench-Verified (69.4%)

Versions: [R2E-Gym](https://github.com/R2E-Gym/R2E-Gym) @ `0d94c4eb9431cd195c55a7ea3abd54006c9a1735` (agent scaffold), `swebench==4.1.0` (official scorer), dataset `princeton-nlp/SWE-bench_Verified`.

```bash
# 1. Run the agent (R2E-Gym scaffold) against the served endpoint — 4 concurrent,
#    seed-42 shuffle, resumable via fixed exp_name. Produces trajectories with
#    ds.instance_id + output_patch per task.
python -m r2egym.agenthub.run.edit runagent_multiple \
  --traj_dir ./traj --exp_name v500 --start_idx 0 --k 500 \
  --dataset R2E-Gym/SWE-Bench-Verified --split test \
  --llm_name openai/qwen3.6-27b --use_fn_calling True \
  --max_steps_absolute 100 --num_workers 4 \
  --use_existing True   # resume support

# 2. Export {instance_id, model_patch} per task -> preds.jsonl (this dir carries ours).

# 3. Official scoring. Docker Hub anonymous pulls rate-limit at ~450/6h — authenticate
#    or expect ~50 bogus "error" instances on a full 500 (check error_instances in the
#    report before believing any aggregate).
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified --predictions_path preds.jsonl \
  --namespace swebench --cache_level env --timeout 1800 \
  --max_workers 8 --run_id deepswe

# 4. Sanitize + rescore (the 62.2% -> 69.4% step, disclosed in the README):
#    R2E task images ship modified build files; the exported git diff carries those
#    image artifacts inside every patch; official git-apply rejects the whole patch and
#    GNU patch's fallback reverse-applies it. The sanitizer strips ONLY root-level
#    build/config hunks, uniformly, and re-runs the harness on affected instances.
bash deepswe-sanitize-rescore.sh
```

Sanity anchors: R2E's own `reward` signal agrees with the official harness within ±2.8% *after* sanitization (8 R2E-only vs 6 official-only); same-model mini-swe-agent reference is 67.8%.

### Is the sanitization legitimate? Verify it yourself

The stripped hunks are provably **image artifacts, not agent work** — three independent fingerprints, all checkable from `preds.jsonl.gz` in this directory:

1. **Incidence**: 43 of 44 sphinx patches contain a `tox.ini` diff block. No agent behavior edits the test-runner config in 98% of unrelated bug-fix tasks.
2. **Byte-identity across different bugs**: the `tox.ini` blocks cluster into ~14 distinct contents that are *byte-identical within sphinx version families* — one block shared verbatim by 8 different tasks (sphinx-9281/9230/9591/9461/9258/9367/9320/9229), another by 7 sphinx-7.x tasks, another by 6 sphinx-8.x tasks. Independent agents solving different bugs cannot produce identical edits; per-version image templates must.
3. **Content**: the modification is `pytest --durations 25` → `pytest -rA --durations 25` — a *test-output reporting flag* (R2E's reward parser needs per-test result lines). It cannot fix a bug and cannot change a test outcome; it is the R2E image authors' own harness plumbing, exported into every patch by end-of-session `git diff` against a worktree the image shipped dirty.

Directionality: the sanitizer strips *only* root-level build/config hunks and *only* when source-file edits remain, uniformly across all 500 instances — it cannot create a solution, only stop the official harness from rejecting a real one over foreign hunks. The independent cross-check is the strongest anchor: R2E's in-container reward (actual test executions, no patch export involved) counted 349 solved; the official harness after sanitization counts 347, with symmetric 8-vs-6 disagreement. The unsanitized 62.2% would contradict test runs that watched those fixes pass — it measures export plumbing, not the model.

## Terminal-Bench 2.1 (48.3%)

Versions: `harbor==0.18.0` (`uv tool install harbor`), dataset `terminal-bench/terminal-bench-2-1` (89 tasks), agent `terminus-2`, k=1, default per-task timeouts (leaderboard validation requires `timeout_multiplier = 1.0`; official rows use k=5).

```bash
# terminus-2's litellm runs HARNESS-side: creds must be in harbor's own process env
# (--ae only reaches the task container).
export OPENAI_API_KEY=local
export OPENAI_API_BASE=http://172.17.0.1:8020/v1
export OPENAI_BASE_URL=http://172.17.0.1:8020/v1
harbor run --dataset terminal-bench/terminal-bench-2-1 --agent terminus-2 \
  --model openai/qwen3.6-27b \
  --ae OPENAI_API_KEY=local --ae OPENAI_API_BASE=http://172.17.0.1:8020/v1 \
  --ae OPENAI_BASE_URL=http://172.17.0.1:8020/v1 \
  --allow-agent-host 172.17.0.1 --n-concurrent 2 --n-attempts 1 \
  --jobs-dir ./jobs --yes
```

Score = `verifier_result.rewards.reward == 1` over each trial's `result.json`; exceptions classified via `exception_info.exception_type` (our split: 43 pass / 17 fail / 27 AgentTimeoutError / 2 harness RuntimeError — `final-c2-outcomes.txt`). Concurrency note: c4 vs c2 changes wall-clock, not score (measured — all 7 c4-pilot timeouts re-timed-out at c2).
