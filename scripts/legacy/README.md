# Legacy launchers

Kept so the configurations in [../../docs/HISTORY.md](../../docs/HISTORY.md) stay reproducible. None of these is the served configuration.

- `serve.sh`: the 2026-07-20 LMCache tier profile on the 0.23 base (LMCache `main` plus the six `patches/lmcache` patches, fp8 KV, V1 runner).
- `serve-nightly.sh`: the 2026-08-15 plain profile on the digest-pinned 0.27 nightly, no patches, no tiers.
- `serve-plain.sh`: the 2026-07-19 plain profile on the 0.23 base with the PR #42603 MTP workaround.

Later generations that are also no longer served: [`../serve-tier-rc4.sh`](../serve-tier-rc4.sh) (LMCache tiers on the 2026-08-21 nightly), [`../serve-nvfp4kv.sh`](../serve-nvfp4kv.sh) and [`../serve-dflash2.sh`](../serve-dflash2.sh) (one-card experiments on that nightly), [`../serve-r134-daily.sh`](../serve-r134-daily.sh) (the two-card configuration on the gittensor checkpoint, frozen as the rollback of the current one).

The current launchers are [`../serve-r156-daily.sh`](../serve-r156-daily.sh) (two cards) and [`../serve-v0280-daily.sh`](../serve-v0280-daily.sh) (one card, and the generic launcher).
