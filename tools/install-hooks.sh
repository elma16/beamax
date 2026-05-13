#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
git -C "$root" config --local core.hooksPath tools/hooks

chmod +x \
  "$root/tools/hooks/check-public-push.sh" \
  "$root/tools/hooks/pre-commit" \
  "$root/tools/hooks/pre-push"

echo "Configured git hooks:"
echo "  core.hooksPath=tools/hooks"
