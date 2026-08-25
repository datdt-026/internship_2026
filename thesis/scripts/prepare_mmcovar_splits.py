#!/usr/bin/env python3
"""Prepare MMCoVaR multimodal splits (text + local image_path)."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocess import clean_text, save_records_jsonl
from src.data.schema import SampleRecord

CSV_PATH = ROOT / "data" / "raw" / "mmcovar" / "MMCoVaR_News_Dataset.csv"
IMG_DIR = ROOT / "data" / "raw" / "mmcovar" / "images"
SPLITS_DIR = ROOT / "data" / "splits_mmcovar"
PROCESSED_DIR = ROOT / "data" / "processed"
STATS_PATH = PROCESSED_DIR / "mmcovar_stats.json"
SOURCE = "mmcovar_news"


def find_image(news_id: str) -> Path | None:
    matches = list(IMG_DIR.glob(f"{news_id}.*"))
    return matches[0] if matches else None


def build_records(*, require_image: bool = True) -> list[SampleRecord]:
    df = pd.read_csv(CSV_PATH)
    records: list[SampleRecord] = []
    skipped_no_text = 0
    skipped_no_image = 0

    for _, row in df.iterrows():
        news_id = str(row["news_id"])
        title = str(row["title"]) if pd.notna(row.get("title")) else ""
        body = str(row["body_text"]) if pd.notna(row.get("body_text")) else ""
        text = clean_text(f"{title}. {body}".strip())
        if not text:
            skipped_no_text += 1
            continue

        # reliability: 1 reliable -> credible, 0 unreliable -> misinfo
        rel = int(row["reliability"])
        label = "credible" if rel == 1 else "misinfo"

        img = find_image(news_id)
        if require_image and img is None:
            skipped_no_image += 1
            continue

        image_path = str(img.relative_to(ROOT)) if img is not None else None
        records.append(
            SampleRecord(
                id=f"mmcovar-{news_id}",
                text=text,
                label=label,
                image_path=image_path,
                platform="news",
                source=SOURCE,
            )
        )

    print(f"skipped_no_text={skipped_no_text} skipped_no_image={skipped_no_image}")
    return records


def stratified_split(records: list[SampleRecord], seed: int = 42):
    labels = [r.label_id() for r in records]
    train_val, test = train_test_split(
        records, test_size=0.15, random_state=seed, stratify=labels
    )
    train_val_labels = [r.label_id() for r in train_val]
    train, val = train_test_split(
        train_val, test_size=0.15 / 0.85, random_state=seed, stratify=train_val_labels
    )
    for name, split_records in [("train", train), ("val", val), ("test", test)]:
        for r in split_records:
            r.split = name
    return {"train": train, "val": val, "test": test}


def stats(splits: dict[str, list[SampleRecord]]) -> dict:
    out: dict = {"source": SOURCE, "splits": {}}
    all_recs = []
    for name, recs in splits.items():
        c = Counter(r.label for r in recs)
        lengths = [len(r.text) for r in recs]
        out["splits"][name] = {
            "n": len(recs),
            "credible": c.get("credible", 0),
            "misinfo": c.get("misinfo", 0),
            "with_image": sum(1 for r in recs if r.image_path),
            "text_len_avg": round(sum(lengths) / len(lengths), 2) if lengths else 0,
        }
        all_recs.extend(recs)
    c = Counter(r.label for r in all_recs)
    out["all"] = {
        "n": len(all_recs),
        "credible": c.get("credible", 0),
        "misinfo": c.get("misinfo", 0),
        "with_image": sum(1 for r in all_recs if r.image_path),
    }
    out["label_map"] = {"reliability_1": "credible", "reliability_0": "misinfo"}
    return out


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing {CSV_PATH}")
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    records = build_records(require_image=True)
    if len(records) < 50:
        raise RuntimeError(
            f"Only {len(records)} multimodal samples. "
            "Run: python scripts/download_mmcovar_images.py"
        )

    splits = stratified_split(records)
    for name, recs in splits.items():
        path = SPLITS_DIR / f"{name}.jsonl"
        save_records_jsonl(recs, path)
        print(f"wrote {path} ({len(recs)})")

    all_path = PROCESSED_DIR / "mmcovar_all.jsonl"
    save_records_jsonl(splits["train"] + splits["val"] + splits["test"], all_path)
    s = stats(splits)
    STATS_PATH.write_text(json.dumps(s, indent=2), encoding="utf-8")
    print(json.dumps(s, indent=2))
    print(f"wrote {all_path}")
    print(f"wrote {STATS_PATH}")
    print("DONE — MMCoVaR multimodal splits ready.")


if __name__ == "__main__":
    main()
