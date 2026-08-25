# Decoding Post-Pandemic Vaccine Hesitancy

Multimodal Transformer pipeline for vaccine misinformation detection (X / TikTok-oriented), USTH Master in ICT — **Do Thanh Dat (M23.ICT.002)**.

**Deadline:** 31/08/2026  
**Survival scope:** text + image, late fusion, small labeled dataset, BERTopic narrative analysis.

Supervisor: Giang Anh Tuan

---

## Status (Step 1 — scaffold)

| Step | Content | Window | Status |
|------|---------|--------|--------|
| 1 | Repo scaffold + configs + stubs | 7–9/08 | **in progress** |
| 2 | Dataset lock-in + preprocess + splits | ~10–12/08 | pending |
| 3 | Text baseline (RoBERTa) + metrics | ~13–15/08 | pending |
| 4 | Late fusion + ablation | ~16–19/08 | pending |
| 5 | BERTopic narratives + thesis writing | ~20–26/08 | pending |
| 6 | Freeze results, slides, submit | ~27–31/08 | pending |

---

## Project layout

```
thesis/
├── configs/default.yaml      # experiment hyperparameters
├── data/
│   ├── raw/                  # original downloads (gitignored)
│   ├── processed/            # cleaned tables
│   └── splits/               # train/val/test jsonl
├── src/
│   ├── data/                 # schema, preprocess, dataset, split
│   ├── models/               # text / image / late fusion
│   ├── utils/                # config, seed, metrics
│   ├── train.py
│   ├── evaluate.py
│   └── topics.py
├── scripts/                  # bootstrap, smoke test, dummy data
├── notebooks/                # EDA / error analysis
├── thesis_docs/              # chapter outlines & figures
├── results/                  # metrics json, plots
└── checkpoints/              # model weights (gitignored)
```

---

## Quick start

```bash
cd thesis
bash scripts/bootstrap_env.sh
source .venv/bin/activate

# Optional: tiny dummy splits (not for real experiments)
python scripts/make_dummy_split.py

# Scaffold entrypoints (no full training yet)
python -m src.train --config configs/default.yaml
python -m src.evaluate --config configs/default.yaml
bash scripts/smoke_test.sh
```

Requires Python 3.10+.

---

## Planned model (after Step 1)

1. **Text-only:** `roberta-base` → CLS → linear head  
2. **Image-only / fusion:** CLIP ViT image encoder (frozen) + late concat MLP  
3. **Metrics:** Accuracy, Precision, Recall, Macro-F1  
4. **Topics:** BERTopic on misinfo texts → hesitancy narrative taxonomy  

---

## Thesis docs

See `thesis_docs/OUTLINE.md` for chapter structure aligned with the survival scope.

---

## Notes for supervisor sync

- Full X+TikTok+Whisper stack is **out of scope** for the 31/08 deadline unless data arrives early.  
- Priority is a **reproducible pipeline + ablation table + honest limitations**.
