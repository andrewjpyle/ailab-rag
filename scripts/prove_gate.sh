#!/usr/bin/env bash
# Proof-of-gate (make red): the regression gate MUST fail on each sabotage, and name the
# failing metric in stderr. Any case that does not fail, or does not name its metric, fails
# this script. A gate you cannot see fail is not a gate.
set -uo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$repo_root"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
out="$tmp/r.json"
empty="$tmp/empty.jsonl"
: >"$empty"

status=0

# $1 human label, $2 expected exit code, $3 metric token expected in stderr, rest: eval args
check() {
  local label="$1" want_code="$2" want_metric="$3"
  shift 3
  local err
  err="$(uv run ailab-eval --output "$out" "$@" 2>&1 >/dev/null)"
  local code=$?
  if [[ "$code" -ne "$want_code" ]]; then
    echo "FAIL [$label]: exit $code, expected $want_code" >&2
    status=1
  elif ! grep -qiE "$want_metric" <<<"$err"; then
    echo "FAIL [$label]: stderr did not name '$want_metric'" >&2
    echo "$err" | sed 's/^/    /' >&2
    status=1
  else
    echo "PASS [$label]: exit $code, named '$want_metric'"
    grep -iE "REGRESSION|error" <<<"$err" | sed 's/^/    /'
  fi
}

echo "== make red: proving the gate fails on each sabotage =="
check "1 broken retrieval (random)"       1 "recall@"          --retriever random --min-recall 0.7
check "1b broken retrieval (top-k 1)"     1 "recall@1"         --top-k 1 --min-recall 0.7
check "2 empty corpus (0 chunks/docs)"    2 "empty"            --corpus "$empty"
check "3 fabricated unanswerable answer"  1 "wrong_refusal"    --force-answer --min-recall 0.7
check "4 unsupported citation"            1 "lexical_support"  --sabotage-citations --min-recall 0.7

if [[ "$status" -eq 0 ]]; then
  echo "== all red cases failed the gate as required =="
else
  echo "== RED PROOF FAILED: a sabotage did not trip the gate ==" >&2
fi
exit "$status"
