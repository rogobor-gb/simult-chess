#!/usr/bin/env bash
# Local commit-gate check (docs/DEVELOPMENT.md §0.5/§7, addendum A11): ruff,
# mypy --strict, and the fast pytest subset. No hosted CI yet (A11) -- this
# is the same script a future GitHub Actions workflow would call.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "==> ruff check"
ruff check .

echo "==> mypy"
mypy

echo "==> pytest -m 'not slow'"
pytest -m "not slow"

echo "==> all checks passed"
