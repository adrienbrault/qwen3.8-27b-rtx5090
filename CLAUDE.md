# CLAUDE.md

This file is the working agreement for this repository, for people and coding agents alike (`AGENTS.md` is a link to it).

This repository is public. It mirrors the serving configuration kept in a private infrastructure repo, and every commit is pushed to GitHub. GitHub keeps serving a commit by its SHA after a history rewrite, so nothing can be unpublished: the checks below run before the commit, never after.

## Never commit

- Credentials in any form: tokens, registry logins, API keys, SSH material, `.netrc` lines, resolved secret-manager values.
- Internal addresses: private (RFC1918) and Tailscale IPs, LAN hostnames other than the `flan` alias, MAC addresses.
- Paths outside the documented host layout under `/srv/qwen5090`: home directories on any machine, macOS or Linux temp directories, assistant scratchpad directories, session identifiers.
- Files that belong to other work. Stage explicit paths. `git add -A` and `git add .` are forbidden; an earlier sweep pushed another session's files into this repo.
- Numbers that were not measured. A projection or an estimate does not go into a results table.

Run `scripts/check-public-hygiene.sh` before every commit. It fails on the patterns above in the staged diff; `--tree` scans the whole checkout.

## README

The README is read top-down by someone who wants to know what is served and how fast it is. Its structure is fixed; a promotion or a new measurement updates it **in place**.

**Top half = the served state only.** The intro, `## Numbers`, `## Served configuration` and `## What the stack is` describe the configuration that is served now (the launcher named under `## Served configuration`), with the numbers measured on it. They never contain:
- before/after comparisons with a previous checkpoint, drafter, pin or date ("was X, now Y", "against the RedHatAI checkpoint it replaced", "the pool until 2026-09-09");
- the story of a promotion, a rollback, a retraction, a re-run or how a question was settled ("promoted with that gap known", the boot-rate ladder behind a pin, the cell a re-run replaced);
- instrument caveats, review findings, per-trial score lists or decompositions of a discrepancy;
- rows measured on another checkpoint, drafter, KV dtype or state dtype. The pin and the sequence limit change only the capacity rows (pool, free VRAM, admitted requests), which must carry the served values.

Those go where they belong: the promotion's row and section in `docs/HISTORY.md`, the round's write-up in `bench/results/`, a ruler reading in `docs/FIDELITY.md`, a trap in `docs/GOTCHAS.md`, and a number from an earlier configuration in `## Other configurations` with that configuration named. The top half may link to them in one clause.

**`## Numbers` layout**, in this order and no longer:
1. One paragraph: the launcher and image, when it was measured, the round and its results directory, the instrument in one sentence, and what "rate" and "aggregate" mean.
2. The decode figure, then one paragraph saying what the solid and dashed lines are.
3. At most five bullets, one or two sentences each.
4. The prefill figure, then at most two sentences.
5. The value / source table, served-state rows only, each with its date and write-up, then the one-line "Also passing" list.

A number that needs a paragraph of explanation belongs in the write-up; the README states it with its conditions and links there.

**Decode metrics**: per-stream decode rate and decode aggregate (the sum of the decode rates of the streams running together) are the headline; time to the first token is separate; a wall-clock figure such as the standard benchmark's output tok/s appears only labelled as one. Never per-stream × N as an aggregate.

**Figures** are drawn by `bench/plot.py` from raw records committed under `bench/results/`; no figure carries a number that is not in this repository. `plot.py` fails when an input file is missing; it never falls back to an older round. A figure from another configuration (the seq-64 figure) sits in `## Other configurations`, not in `## Numbers`.

The lower sections (`## How the numbers are measured`, `## Hardware`, `## Quick start`, `## Other configurations`, `## Findings that transfer`) carry method detail and may name earlier rounds and configurations, in the same declarative prose.

## Prose in README and docs

- Declarative sentences. Each states what was measured, when, on which configuration, where the raw output is, and what it means.
- No evaluative or promotional words: nothing is "blazing", "robust", "battle-tested", "seamless", "powerful", "gold", "huge", "simple" or "just". No exclamation marks, no rhetorical questions, no "we're excited".
- No filler: no "note that", "it is worth noting", "importantly", "in other words", "as mentioned above", and no closing paragraph that restates the section.
- One paragraph is one line in the source; no manual wrapping.
- Headings name the finding, not the activity: "fp8 KV costs +0.13 points of perplexity", not "KV cache investigation".
- Every number carries a date and a results directory. A comparison names both arms and the instrument.
- A rejected or reversed result is written up the same way as a success, with the number that rejected it.

## Attribution

A patch, flag or technique taken from a PR, issue, repo or paper is credited in `THIRD_PARTY.md` in the same commit that adds it, whether code or idea.

## Sync with the private repo

The launchers here (`scripts/serve*.sh`) and the private repo's `flan/launch-*.sh` describe the same served configuration and change in the same session. Each experiment gets its own file in `bench/results/`, named after it, with the heading naming its results directory; `bench/RESULTS.md` is the index of links to them, newest first. A new result means a new file plus one index line — never prose appended to the index.

## Recipe card

The README's card is `docs/img/recipe-card-<first 8 hex of its SHA-256>.png`. Each refresh adds the new file under its new name, removes the old one and updates README line 3 in the same commit: GitHub's image cache keeps serving an unchanged URL for minutes after a push.
