# Recommended sampling (T=1.0) vs the T=0.6 override: the override is kept (2026-08-27, `results/2026-08-27-recsettings`)

[← all results](../RESULTS.md)

Every Qwen3.8 checkpoint's generation_config recommends T=1.0 / top_p 0.95 / top_k 20, while this stack overrides temperature to 0.6 on evidence inherited from the Qwen3.6 era. The retest on 3.8 used a pre-registered decision rule: a new T=1.0 arm for GSM8K (rescored) and IFEval on four NVFP4 checkpoints against the existing T=0.6 baselines, plus paired same-session tool-eval 69x4 at both temperatures. Cross-day tool-eval cannot resolve <3 pts, as the noise-floor section above shows.

**GSM8K ties everywhere** (maximum delta -0.8, about 1.2 sigma; truncation counters ruled out artifacts). Paired tool-evals all tie, with signs flipping per checkpoint (-1.3 / +1.2 / +1.5), so the 3.6-era result that T=0.6 wins tools by about 3 does not reproduce on 3.8. The one sign-stable effect is that IFEval drops at T=1.0 in all 8 readings (inst- and prompt-level, four checkpoints; mean about -3, for example gittensor inst-loose 69.4 -> 65.4). T=1.0 buys nothing measured here and costs instruction adherence, so the serve scripts keep the T=0.6 override, now on same-generation, noise-floor-aware evidence.
