#!/usr/bin/env python3
"""Classical TF-IDF + Logistic Regression text baseline.

Usage (from thesis/):
  PYTHONPATH=. python scripts/run_sklearn_baseline.py
  PYTHONPATH=. python scripts/run_sklearn_baseline.py --splits-dir data/splits_mmcovar --tag mmcovar
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.preprocess import load_records_jsonl
from src.utils.metrics import compute_classification_metrics, format_classification_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TF-IDF + LogisticRegression baseline")
    p.add_argument("--splits-dir", type=str, default="data/splits")
    p.add_argument("--output-dir", type=str, default="results")
    p.add_argument("--tag", type=str, default="constraint", help="Name tag for output files")
    p.add_argument("--max-features", type=int, default=50000)
    p.add_argument("--ngram-max", type=int, default=2)
    p.add_argument("--C", type=float, default=1.0)
    return p.parse_args()


def load_xy(path: Path) -> tuple[list[str], list[int]]:
    records = load_records_jsonl(path)
    texts = [r.text for r in records]
    labels = [r.label_id() for r in records]
    return texts, labels


def main() -> None:
    args = parse_args()
    splits = Path(args.splits_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train = load_xy(splits / "train.jsonl")
    x_val, y_val = load_xy(splits / "val.jsonl")
    x_test, y_test = load_xy(splits / "test.jsonl")

    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=args.max_features,
                    ngram_range=(1, args.ngram_max),
                    lowercase=True,
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=args.C,
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                ),
            ),
        ]
    )

    print(f"[sklearn] fitting on {len(x_train)} train samples ({args.tag})...")
    pipe.fit(x_train, y_train)

    results: dict = {
        "status": "ok",
        "model": "tfidf_logreg",
        "tag": args.tag,
        "splits_dir": str(splits),
        "hyperparams": {
            "max_features": args.max_features,
            "ngram_range": [1, args.ngram_max],
            "C": args.C,
            "class_weight": "balanced",
            "solver": "liblinear",
        },
        "n_train": len(x_train),
        "n_val": len(x_val),
        "n_test": len(x_test),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }

    for split_name, xs, ys in [("val", x_val, y_val), ("test", x_test, y_test)]:
        pred = pipe.predict(xs)
        metrics = compute_classification_metrics(ys, pred, average="macro")
        report = format_classification_report(ys, pred, target_names=["credible", "misinfo"])
        results[f"{split_name}_metrics"] = metrics
        results[f"{split_name}_report"] = report
        print(f"[sklearn] {split_name}: {json.dumps(metrics)}")
        print(report)

    out_path = out_dir / f"sklearn_baseline_{args.tag}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[sklearn] wrote {out_path}")


if __name__ == "__main__":
    main()
