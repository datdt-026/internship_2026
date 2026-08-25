# Data directory

| Path | Purpose |
|------|---------|
| `raw/` | Original downloads / exports (not committed) |
| `processed/` | Cleaned tables before splitting |
| `splits/` | `train.jsonl`, `val.jsonl`, `test.jsonl` |

## Sample schema (JSONL)

```json
{
  "id": "x-123",
  "text": "Vaccines reduce severe disease risk.",
  "label": "credible",
  "image_path": "data/raw/images/x-123.jpg",
  "platform": "x",
  "source": "dataset-name",
  "split": "train"
}
```

Labels: `credible` | `misinfo` (aliases normalized in `src/data/schema.py`).

## Current primary source (Step 2)

**CONSTRAINT COVID-19 Fake News** via Hugging Face:
`nanyy1025/covid_fake_news` (Patwa et al.)

```bash
# from thesis/
source .venv/bin/activate
python scripts/download_constraint_hf.py
```

Files land in:
`data/raw/constraint_covid_fake_news/{train,validation,test,all}.csv`

Columns: `id`, `tweet`, `label` (`real` / `fake`).

Prepare processed splits:

```bash
python scripts/prepare_constraint_splits.py
```

Outputs:
- `data/processed/constraint_all.jsonl`
- `data/processed/constraint_stats.json`
- `data/splits/{train,val,test}.jsonl` with labels `credible` / `misinfo`

Optional (needs HF account / higher rate limits):
```bash
export HF_TOKEN=hf_xxx
```

## MMCoVaR multimodal (Step 4)

```bash
python scripts/download_mmcovar_images.py --workers 12
python scripts/prepare_mmcovar_splits.py
python -m src.train --config configs/mmcovar.yaml --modality fusion
# ablation:
bash scripts/run_mmcovar_ablation.sh
```

Splits: `data/splits_mmcovar/` · Stats: `data/processed/mmcovar_stats.json`
