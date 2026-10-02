#!/usr/bin/env bash
# Part 4: regenerate the fix-loop BEFORE run (git tag fixloop-before) and AFTER run (working tree)
# from the raw captures, score both with the same scorer, write reports/fixloop/before_after.md.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
rm -rf /tmp/scan2plan_before && git worktree add -f /tmp/scan2plan_before fixloop-before >/dev/null
for d in c00a170fe1 1a8384c3f6 c7d28f72c6; do
  (cd /tmp/scan2plan_before && PYTHONPATH=. "$OLDPWD/$PY" -m scan2plan run "$OLDPWD/data/$d" \
      --out "$OLDPWD/out/fixloop/before/$d/lidar" --no-damage --quiet)
  $PY -m scan2plan run data/$d --out out/fixloop/after/$d/lidar --no-damage --quiet
done
git worktree remove --force /tmp/scan2plan_before
$PY -m bench.fixloop_compare out/fixloop/before out/fixloop/after
