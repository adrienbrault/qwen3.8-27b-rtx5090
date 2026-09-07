# Quality checks of the sweep config, tier included: passed (2026-08-31, `results/2026-08-31-r133-dflash-quality`)

[← all results](../RESULTS.md)

The DFlash2-fp8-TP=2 sweep config measured above holds up on quality: **tool-eval 90.2 ± 1.5** (daily: 90.0 ± 1.4), GSM8K T=0 0.8417 ± .034 (daily: 0.842), zero needle failures under an 8-way 45K-token flood, and deep-100K decode 130.8 (+13% over the daily). At that depth its 17.8 s TTFT implies about 5.6K t/s prefill, faster than single-GPU, so the TP=2 prefill tax is mid-range only (−15% @30K, +14% @100K). The rule that DFlash2 cannot have KV tiers turned out to be an artifact of the old LMCache connector: the native OffloadingConnector boots clean under DFlash2+TP=2, serves a correct post-restart revisit from disk, and still decodes at 239.8 c1. The result is a complete serving candidate: every speed cell, matched quality, 711K pool, disk tier, 262K context.
