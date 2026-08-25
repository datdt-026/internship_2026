#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Recreate if missing OR broken (e.g. empty dir left by a failed earlier run).
if [[ ! -f .venv/bin/activate ]]; then
  echo "[bootstrap] Creating virtualenv at $ROOT/.venv"
  rm -rf .venv
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo
echo "Environment ready. Activate with:"
echo "  source $ROOT/.venv/bin/activate"
echo "Then from repo root:"
echo "  python -m src.train --config configs/default.yaml"
echo "  bash scripts/smoke_test.sh"
