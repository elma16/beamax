#!/usr/bin/env bash
set -euo pipefail

remote_name="${1:-}"
remote_url="${2:-}"

blocked_ref='refs/heads/private/'
blocked_paths='(^|/)(examples/thesis|examples/learned|examples/private|data|checkpoints|outputs|plots|plots_focusing|wandb|cache|profiler|htmlcov|tests/test-data)(/|$)'
zero_sha='0000000000000000000000000000000000000000'
current_ref="$(git symbolic-ref --quiet HEAD 2>/dev/null || true)"
current_branch="${current_ref#refs/heads/}"

is_public_remote=false
if [[ "$remote_name" == "upstream" || "$remote_url" == *"github.com:elma16/beamax.git"* || "$remote_url" == *"github.com/elma16/beamax.git"* ]]; then
  is_public_remote=true
fi

if [[ "$current_ref" == "$blocked_ref"* ]]; then
  echo "blocked push: current branch '$current_branch' is local-only; switch to main before publishing" >&2
  exit 1
fi

blocked_commit_with_private_paths() {
  local local_sha="$1"
  local remote_sha="$2"
  local range

  if [[ "$local_sha" == "$zero_sha" ]]; then
    return 1
  fi

  if [[ "$remote_sha" == "$zero_sha" ]]; then
    range="$local_sha"
  else
    range="$remote_sha..$local_sha"
  fi

  while read -r commit_sha; do
    if git diff-tree --no-commit-id --name-only -r "$commit_sha" | grep -E "$blocked_paths" >/dev/null; then
      echo "$commit_sha"
      return 0
    fi
  done < <(git rev-list "$range")

  return 1
}

while read -r local_ref local_sha remote_ref remote_sha; do
  if [[ -z "${local_ref:-}" ]]; then
    continue
  fi

  if [[ "$local_ref" == "$blocked_ref"* || "$remote_ref" == "$blocked_ref"* ]]; then
    echo "blocked push: private branch refs must stay local ($local_ref -> $remote_ref)" >&2
    exit 1
  fi

  if [[ "$is_public_remote" == true ]]; then
    if [[ "$local_ref" != "refs/heads/main" || "$remote_ref" != "refs/heads/main" ]]; then
      echo "blocked push: public upstream only accepts refs/heads/main -> refs/heads/main ($local_ref -> $remote_ref)" >&2
      exit 1
    fi

    if blocked_commit="$(blocked_commit_with_private_paths "$local_sha" "$remote_sha")"; then
      echo "blocked push: private/data paths are present in commit $blocked_commit" >&2
      exit 1
    fi
  fi
done
