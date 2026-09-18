#!/usr/bin/env python3
"""Prepare MMCoVaR splits with NO shared publishers across train/val/test."""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocess import clean_text, save_records_jsonl
from src.data.schema import SampleRecord

CSV_PATH = ROOT / "data" / "raw" / "mmcovar" / "MMCoVaR_News_Dataset.csv"
IMG_DIR = ROOT / "data" / "raw" / "mmcovar" / "images"
SPLITS_DIR = ROOT / "data" / "splits_mmcovar_pubdisjoint"
PROCESSED_DIR = ROOT / "data" / "processed"
STATS_PATH = PROCESSED_DIR / "mmcovar_pubdisjoint_stats.json"
SOURCE = "mmcovar_news_pubdisjoint"
SEED = 42


def find_image(news_id: str) -> Path | None:
    matches = list(IMG_DIR.glob(f"{news_id}.*"))
    return matches[0] if matches else None


def normalize_publisher(row: pd.Series) -> str:
    for col in ("publisher", "Publisher"):
        if col in row.index and pd.notna(row[col]):
            val = str(row[col]).strip()
            if val and val.lower() not in {"nan", "none"}:
                return val
    return f"UNKNOWN_{row['news_id']}"


def build_by_publisher() -> dict[str, list[SampleRecord]]:
    df = pd.read_csv(CSV_PATH)
    by_pub: dict[str, list[SampleRecord]] = defaultdict(list)
    for _, row in df.iterrows():
        news_id = str(row["news_id"])
        title = str(row["title"]) if pd.notna(row.get("title")) else ""
        body = str(row["body_text"]) if pd.notna(row.get("body_text")) else ""
        text = clean_text(f"{title}. {body}".strip())
        if not text:
            continue
        img = find_image(news_id)
        if img is None:
            continue
        rel = int(row["reliability"])
        label = "credible" if rel == 1 else "misinfo"
        pub = normalize_publisher(row)
        rec = SampleRecord(
            id=f"mmcovar-{news_id}",
            text=text,
            label=label,
            image_path=str(img.relative_to(ROOT)),
            platform="news",
            source=SOURCE,
        )
        by_pub[pub].append(rec)
    return by_pub


def assign_publishers(by_pub: dict[str, list[SampleRecord]], seed: int = SEED) -> dict[str, set[str]]:
    """Assign each publisher to exactly one split; target ~70/15/15 articles."""
    rng = random.Random(seed)
    pubs = sorted(by_pub.keys(), key=lambda p: (-len(by_pub[p]), rng.random()))
    total = sum(len(v) for v in by_pub.values())
    target_train = int(round(total * 0.70))
    target_val = int(round(total * 0.15))

    assignment: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    counts = {"train": 0, "val": 0, "test": 0}

    for pub in pubs:
        n = len(by_pub[pub])
        if counts["train"] < target_train:
            assignment["train"].add(pub)
            counts["train"] += n
        elif counts["val"] < target_val:
            assignment["val"].add(pub)
            counts["val"] += n
        else:
            assignment["test"].add(pub)
            counts["test"] += n

    for split in ("val", "test"):
        if not assignment[split] and assignment["train"]:
            donor = min(assignment["train"], key=lambda p: len(by_pub[p]))
            assignment["train"].remove(donor)
            assignment[split].add(donor)
    return assignment


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(CSV_PATH)
    by_pub = build_by_publisher()
    assignment = assign_publishers(by_pub, seed=SEED)

    assert not (assignment["train"] & assignment["val"])
    assert not (assignment["train"] & assignment["test"])
    assert not (assignment["val"] & assignment["test"])

    splits: dict[str, list[SampleRecord]] = {"train": [], "val": [], "test": []}
    for split, pubs in assignment.items():
        for pub in pubs:
            for r in by_pub[pub]:
                r.split = split
                splits[split].append(r)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, recs in splits.items():
        path = SPLITS_DIR / f"{name}.jsonl"
        save_records_jsonl(recs, path)
        print(f"wrote {path} n={len(recs)} publishers={len(assignment[name])}")

    out = {
        "protocol": "publisher_disjoint",
        "seed": SEED,
        "n_articles_total": sum(len(v) for v in splits.values()),
        "n_publishers": {s: len(assignment[s]) for s in assignment},
        "publishers": {s: sorted(assignment[s]) for s in assignment},
        "overlap_publishers_train_test": 0,
        "splits": {},
    }
    for name, recs in splits.items():
        c = Counter(r.label for r in recs)
        out["splits"][name] = {
            "n": len(recs),
            "credible": c.get("credible", 0),
            "misinfo": c.get("misinfo", 0),
            "n_publishers": len(assignment[name]),
        }
    STATS_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"n_publishers": out["n_publishers"], "splits": out["splits"]}, indent=2))
    print(f"wrote {STATS_PATH}")
    print("DONE")


if __name__ == "__main__":
    main()
