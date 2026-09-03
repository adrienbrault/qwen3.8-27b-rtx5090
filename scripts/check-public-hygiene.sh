#!/usr/bin/env bash
# Fails when the staged diff (default) or the whole checkout (--tree) contains something that must not reach the
# public repo: private or Tailscale addresses, local machine paths, assistant scratchpad paths, session identifiers,
# credential-shaped strings. CLAUDE.md lists the rules. Run before every commit; nothing can be unpublished afterwards.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
SELF=scripts/check-public-hygiene.sh
PATS=(
  '(^|[^0-9.])10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'
  '(^|[^0-9.])192\.168\.[0-9]{1,3}\.[0-9]{1,3}'
  '(^|[^0-9.])172\.(1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3}'
  '(^|[^0-9.])100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.[0-9]{1,3}\.[0-9]{1,3}'
  '/Users/[A-Za-z]'
  '/home/[a-z]'
  '/private/tmp/'
  'claude-501'
  '/scratchpad/'
  'claude\.ai/code/session_'
  '(^|[^A-Za-z0-9])ghp_[A-Za-z0-9]{30,}'
  '(^|[^A-Za-z0-9])github_pat_[A-Za-z0-9_]{20,}'
  '(^|[^A-Za-z0-9])gho_[A-Za-z0-9]{30,}'
  '(^|[^A-Za-z0-9])hf_[A-Za-z0-9]{30,}'
  '(^|[^A-Za-z0-9])sk-[A-Za-z0-9_-]{30,}'
  'AKIA[0-9A-Z]{16}'
  'BEGIN [A-Z ]*PRIVATE KEY'
  'xox[bpa]-[A-Za-z0-9-]{20,}'
)
RE=$(IFS='|'; echo "${PATS[*]}")
# Exempt: the docker bridge gateway (172.17.0.1 is the same on every host), and lines carrying a hygiene-ok marker
# with a reason (fictional addresses inside frozen benchmark prompts).
EXEMPT='172\.17\.0\.1|hygiene-ok'
hits=0
if [ "${1:-}" = "--tree" ]; then
  while IFS= read -r -d '' f; do
    [ "$f" = "$SELF" ] || [ "$f" = "CLAUDE.md" ] && continue
    grep -aInE "$RE" -- "$f" | grep -avE "$EXEMPT" | sed "s|^|$f:|" | grep -a . && hits=1
  done < <(git ls-files -z)
else
  # added lines of the staged diff, tagged with their file
  out=$(git diff --cached -U0 --diff-filter=AM -- . ":!$SELF" ":!CLAUDE.md" \
    | awk '/^\+\+\+ b\//{f=substr($0,7)} /^\+/ && !/^\+\+\+/{print f": "substr($0,2)}' \
    | grep -aE "$RE" | grep -avE "$EXEMPT")
  [ -n "$out" ] && { echo "$out"; hits=1; }
fi
if [ "$hits" = 1 ]; then echo "check-public-hygiene: FAIL (see lines above)"; exit 1; fi
echo "check-public-hygiene: OK"
