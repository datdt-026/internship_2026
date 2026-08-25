#!/usr/bin/env python3
"""Prepare CONSTRAINT COVID fake news CSVs into processed + train/val/test JSONL."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocess import clean_text, save_records_jsonl
from src.data.schema import SampleRecord, normalize_label

RAW_DIR = ROOT / "data" / "raw" / "constraint_covid_fake_news"
PROCESSED_DIR = ROOT / "data" / "processed"
SPLITS_DIR = ROOT / "data" / "splits"
STATS_PATH = PROCESSED_DIR / "constraint_stats.json"

# HF split name -> canonical split name used by the project
SPLIT_FILES = {
    "train": "train.csv",
    "val": "validation.csv",
    "test": "test.csv",
}
SOURCE = "constraint_covid_fake_news"
PLATFORM = "social"  # mixed Twitter/Facebook/etc. in CONSTRAINT


def load_split_csv(path: Path, split: str) -> list[SampleRecord]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python scripts/download_constraint_hf.py"
        )
    df = pd.read_csv(path)
    required = {"id", "tweet", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    records: list[SampleRecord] = []
    skipped = 0
    for _, row in df.iterrows():
        text = clean_text(str(row["tweet"]) if pd.notna(row["tweet"]) else "")
        if not text:
            skipped += 1
            continue
        label = normalize_label(row["label"])
        records.append(
            SampleRecord(
                id=f"constraint-{split}-{row['id']}",
                text=text,
                label=label,
                image_path=None,
                platform=PLATFORM,
                source=SOURCE,
                split=split,
            )
        )
    if skipped:
        print(f"[{split}] skipped empty texts: {skipped}")
    return records


def compute_stats(splits: dict[str, list[SampleRecord]]) -> dict:
    stats: dict = {"source": SOURCE, "splits": {}}
    all_records: list[SampleRecord] = []
    for name, records in splits.items():
        lengths = [len(r.text) for r in records]
        counts = Counter(r.label for r in records)
        split_stats = {
            "n": len(records),
            "credible": counts.get("credible", 0),
            "misinfo": counts.get("misinfo", 0),
            "with_image": sum(1 for r in records if r.image_path),
            "text_len_avg": round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
            "text_len_min": min(lengths) if lengths else 0,
            "text_len_max": max(lengths) if lengths else 0,
        }
        stats["splits"][name] = split_stats
        all_records.extend(records)

    all_counts = Counter(r.label for r in all_records)
    all_lengths = [len(r.text) for r in all_records]
    stats["all"] = {
        "n": len(all_records),
        "credible": all_counts.get("credible", 0),
        "misinfo": all_counts.get("misinfo", 0),
        "with_image": sum(1 for r in all_records if r.image_path),
        "text_len_avg": round(sum(all_lengths) / len(all_lengths), 2) if all_lengths else 0.0,
        "text_len_min": min(all_lengths) if all_lengths else 0,
        "text_len_max": max(all_lengths) if all_lengths else 0,
    }
    stats["label_map"] = {"real": "credible", "fake": "misinfo"}
    return stats


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    splits: dict[str, list[SampleRecord]] = {}
    for split, filename in SPLIT_FILES.items():
        records = load_split_csv(RAW_DIR / filename, split)
        splits[split] = records
        print(f"[{split}] loaded {len(records)} records from {filename}")

    # Save individual splits
    for split, records in splits.items():
        out = SPLITS_DIR / f"{split}.jsonl"
        save_records_jsonl(records, out)
        print(f"wrote {out} ({len(records)})")

    # Save combined processed file
    all_records = splits["train"] + splits["val"] + splits["test"]
    processed_path = PROCESSED_DIR / "constraint_all.jsonl"
    save_records_jsonl(all_records, processed_path)
    print(f"wrote {processed_path} ({len(all_records)})")

    stats = compute_stats(splits)
    STATS_PATH.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {STATS_PATH}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print("DONE — Step 2 splits ready.")


if __name__ == "__main__":
    main()
