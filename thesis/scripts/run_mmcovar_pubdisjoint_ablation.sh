#!/usr/bin/env bash
# Publisher-disjoint MMCoVaR ablation
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

CONFIG="${1:-configs/mmcovar_pubdisjoint.yaml}"

echo "[pubdisjoint] TEXT"
python -m src.train --config "$CONFIG" --modality text
echo "[pubdisjoint] IMAGE"
python -m src.train --config "$CONFIG" --modality image
echo "[pubdisjoint] FUSION"
python -m src.train --config "$CONFIG" --modality fusion

python - <<'PY'
import json
from pathlib import Path
rows = []
outdir = Path("results/pubdisjoint")
for m in ["text", "image", "fusion"]:
    p = outdir / f"{m}_results.json"
    if p.exists():
        d = json.loads(p.read_text())
        tm = d["test_metrics"]
        rows.append({"modality": m, **{k: tm[k] for k in ("accuracy", "precision", "recall", "f1")}})
print("\n=== Publisher-disjoint ablation (test) ===")
for r in rows:
    print(f"{r['modality']:8} f1={r['f1']:.4f} acc={r['accuracy']:.4f}")
summary = {
    "protocol": "publisher_disjoint",
    "results": rows,
}
(outdir / "mmcovar_pubdisjoint_ablation_summary.json").write_text(json.dumps(summary, indent=2))
print("wrote", outdir / "mmcovar_pubdisjoint_ablation_summary.json")
PY
