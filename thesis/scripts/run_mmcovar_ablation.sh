#!/usr/bin/env bash
# Run MMCoVaR ablation: text / image / fusion
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="${1:-configs/mmcovar.yaml}"

echo "[ablation] TEXT"
python -m src.train --config "$CONFIG" --modality text
echo "[ablation] IMAGE"
python -m src.train --config "$CONFIG" --modality image
echo "[ablation] FUSION"
python -m src.train --config "$CONFIG" --modality fusion

python - <<'PY'
import json
from pathlib import Path
rows = []
for m in ["text", "image", "fusion"]:
    p = Path("results") / f"{m}_results.json"
    if p.exists():
        d = json.loads(p.read_text())
        tm = d["test_metrics"]
        rows.append((m, tm["accuracy"], tm["precision"], tm["recall"], tm["f1"]))
print("\n=== Ablation (test) ===")
print(f"{'modality':10} {'acc':>8} {'prec':>8} {'rec':>8} {'f1':>8}")
for r in rows:
    print(f"{r[0]:10} {r[1]:8.4f} {r[2]:8.4f} {r[3]:8.4f} {r[4]:8.4f}")
Path("results/mmcovar_ablation_summary.json").write_text(
    json.dumps([{"modality": a, "accuracy": b, "precision": c, "recall": d, "f1": e} for a,b,c,d,e in rows], indent=2)
)
print("wrote results/mmcovar_ablation_summary.json")
PY
