#!/usr/bin/env python3
"""Check publisher overlap across MMCoVaR train/val/test splits.

Joins split JSONL ids (mmcovar-{news_id}) with the original CSV publisher field.
Used to answer defense questions about source-level / publisher leakage.

Usage (from thesis/):
  PYTHONPATH=. python scripts/check_publisher_overlap.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.preprocess import load_records_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MMCoVaR publisher overlap check")
    p.add_argument(
        "--csv",
        type=str,
        default="data/raw/mmcovar/MMCoVaR_News_Dataset.csv",
    )
    p.add_argument("--splits-dir", type=str, default="data/splits_mmcovar")
    p.add_argument("--output", type=str, default="results/publisher_overlap_mmcovar.json")
    return p.parse_args()


def news_id_from_sample_id(sample_id: str) -> str:
    # mmcovar-937 -> 937
    if sample_id.startswith("mmcovar-"):
        return sample_id[len("mmcovar-") :]
    return sample_id


def normalize_publisher(row: pd.Series) -> str:
    for col in ("publisher", "Publisher"):
        if col in row.index and pd.notna(row[col]):
            val = str(row[col]).strip()
            if val and val.lower() not in {"nan", "none"}:
                return val
    return "UNKNOWN"


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    splits_dir = Path(args.splits_dir)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    id_to_pub: dict[str, str] = {}
    for _, row in df.iterrows():
        nid = str(row["news_id"])
        id_to_pub[nid] = normalize_publisher(row)

    split_ids: dict[str, list[str]] = {}
    split_pubs: dict[str, list[str]] = {}
    missing_meta = 0

    for split in ("train", "val", "test"):
        records = load_records_jsonl(splits_dir / f"{split}.jsonl")
        ids: list[str] = []
        pubs: list[str] = []
        for r in records:
            nid = news_id_from_sample_id(r.id)
            ids.append(nid)
            pub = id_to_pub.get(nid)
            if pub is None:
                missing_meta += 1
                pub = "UNKNOWN"
            pubs.append(pub)
        split_ids[split] = ids
        split_pubs[split] = pubs

    sets = {s: set(split_pubs[s]) for s in split_pubs}
    # Exclude UNKNOWN from overlap stats if present
    for s in sets:
        sets[s].discard("UNKNOWN")

    train, val, test = sets["train"], sets["val"], sets["test"]
    train_test = train & test
    train_val = train & val
    val_test = val & test
    all_three = train & val & test

    # Article-level: fraction of test articles whose publisher also appears in train
    test_articles = split_pubs["test"]
    n_test = len(test_articles)
    n_test_leaky = sum(1 for p in test_articles if p in train and p != "UNKNOWN")
    leaky_rate = n_test_leaky / n_test if n_test else 0.0

    # Per-publisher counts in test that overlap train
    test_overlap_counts = Counter(p for p in test_articles if p in train and p != "UNKNOWN")

    summary = {
        "status": "ok",
        "csv": str(csv_path),
        "splits_dir": str(splits_dir),
        "n_articles": {s: len(split_ids[s]) for s in split_ids},
        "n_unique_publishers": {s: len(sets[s]) for s in sets},
        "n_publishers_total_unique": len(train | val | test),
        "overlap_publishers": {
            "train_and_test": sorted(train_test),
            "n_train_and_test": len(train_test),
            "train_and_val": sorted(train_val),
            "n_train_and_val": len(train_val),
            "val_and_test": sorted(val_test),
            "n_val_and_test": len(val_test),
            "all_three_splits": sorted(all_three),
            "n_all_three": len(all_three),
        },
        "jaccard": {
            "train_test": jaccard(train, test),
            "train_val": jaccard(train, val),
            "val_test": jaccard(val, test),
        },
        "test_article_leakage": {
            "n_test_articles": n_test,
            "n_test_articles_whose_publisher_in_train": n_test_leaky,
            "fraction": round(leaky_rate, 4),
            "interpretation": (
                "High fraction means many test articles come from publishers also seen in train "
                "(source-style leakage risk under publisher-reliability labels)."
            ),
        },
        "top_overlapping_publishers_in_test": test_overlap_counts.most_common(15),
        "missing_csv_metadata_rows": missing_meta,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== MMCoVaR publisher overlap ===")
    print(f"Unique publishers: train={len(train)} val={len(val)} test={len(test)}")
    print(f"Publishers in BOTH train and test: {len(train_test)}")
    print(f"Jaccard(train, test) publishers: {summary['jaccard']['train_test']:.3f}")
    print(
        f"Test articles whose publisher appears in train: "
        f"{n_test_leaky}/{n_test} ({100 * leaky_rate:.1f}%)"
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
