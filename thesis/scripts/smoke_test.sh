#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

python - <<'PY'
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.schema import normalize_label, SampleRecord
from src.data.split import stratified_split, split_stats

cfg = load_config("configs/default.yaml")
set_seed(cfg["project"]["seed"])

records = [
    SampleRecord(id="1", text="Vaccines save lives and reduce severe disease.", label="credible"),
    SampleRecord(id="2", text="mRNA vaccines rewrite your DNA permanently.", label="misinfo"),
    SampleRecord(id="3", text="Booster doses improve protection against variants.", label="credible"),
    SampleRecord(id="4", text="Vaccines contain microchips for tracking.", label="misinfo"),
    SampleRecord(id="5", text="Clinical trials monitor vaccine safety signals.", label="credible"),
    SampleRecord(id="6", text="COVID vaccines make people magnetic.", label="misinfo"),
]
for r in records:
    assert normalize_label(r.label) in {"credible", "misinfo"}

# Tiny stratified split smoke (may warn if classes tiny; still validates API)
splits = stratified_split(records, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=42)
print("split_stats:", split_stats(splits))
print("SMOKE OK")
PY

python -m src.train --config configs/default.yaml
python -m src.evaluate --config configs/default.yaml
python -m src.topics --config configs/default.yaml

echo "All smoke checks passed."
