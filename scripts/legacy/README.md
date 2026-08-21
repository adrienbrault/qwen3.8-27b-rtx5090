# Legacy launchers

Kept for reproducibility of the generations in [../../docs/HISTORY.md](../../docs/HISTORY.md). None of these is the daily.

- `serve.sh` — the 2026-07-20 tier profile on the 0.23 base (LMCache `main` + the six `patches/lmcache` patches, fp8 KV, V1 runner).
- `serve-nightly.sh` — the 2026-08-15 plain profile on the digest-pinned 0.27 nightly, no patches, no tiers.
- `serve-plain.sh` — the 2026-07-19 plain profile on the 0.23 base with the PR #42603 MTP workaround.

The current launcher is [`../serve-tier-rc4.sh`](../serve-tier-rc4.sh).
