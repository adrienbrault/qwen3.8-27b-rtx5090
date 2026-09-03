# CLAUDE.md

This repository is public. It mirrors the serving configuration kept in a private infrastructure repo, and every commit is pushed to GitHub. GitHub keeps serving a commit by its SHA after a history rewrite, so nothing can be unpublished: the checks below run before the commit, never after.

## Never commit

- Credentials in any form: tokens, registry logins, API keys, SSH material, `.netrc` lines, resolved secret-manager values.
- Internal addresses: private (RFC1918) and Tailscale IPs, LAN hostnames other than the `flan` alias, MAC addresses.
- Paths outside the documented host layout under `/srv/qwen5090`: home directories on any machine, macOS or Linux temp directories, assistant scratchpad directories, session identifiers.
- Files that belong to other work. Stage explicit paths. `git add -A` and `git add .` are forbidden; an earlier sweep pushed another session's files into this repo.
- Numbers that were not measured. A projection or an estimate does not go into a results table.

Run `scripts/check-public-hygiene.sh` before every commit. It fails on the patterns above in the staged diff; `--tree` scans the whole checkout.

## Prose in README and docs

- Declarative sentences. Each states what was measured, when, on which configuration, where the raw output is, and what it means.
- No evaluative or promotional words: nothing is "blazing", "robust", "battle-tested", "seamless", "powerful", "gold", "huge", "simple" or "just". No exclamation marks, no rhetorical questions, no "we're excited".
- No filler: no "note that", "it is worth noting", "importantly", "in other words", "as mentioned above", and no closing paragraph that restates the section.
- Headings name the finding, not the activity: "fp8 KV costs +0.13 points of perplexity", not "KV cache investigation".
- Every number carries a date and a results directory. A comparison names both arms and the instrument.
- A rejected or reversed result is written up the same way as a success, with the number that rejected it.

## Attribution

A patch, flag or technique taken from a PR, issue, repo or paper is credited in `THIRD_PARTY.md` in the same commit that adds it, whether code or idea.

## Sync with the private repo

The launchers here (`scripts/serve*.sh`) and the private repo's `flan/launch-*.sh` describe the same served configuration and change in the same session. Results land in `bench/RESULTS.md` newest first, with the section naming its results directory.
