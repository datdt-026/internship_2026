from __future__ import annotations

from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

from .preprocess import save_records_jsonl
from .schema import SampleRecord


def stratified_split(
    records: list[SampleRecord],
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[SampleRecord]]:
    """Create stratified train/val/test splits."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    labels = [r.label_id() for r in records]
    train_val, test = train_test_split(
        records,
        test_size=test_ratio,
        random_state=seed,
        stratify=labels,
    )
    relative_val = val_ratio / (train_ratio + val_ratio)
    train_val_labels = [r.label_id() for r in train_val]
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        random_state=seed,
        stratify=train_val_labels,
    )

    for split_name, split_records in [("train", train), ("val", val), ("test", test)]:
        for r in split_records:
            r.split = split_name

    return {"train": train, "val": val, "test": test}


def save_splits(
    splits: dict[str, list[SampleRecord]],
    splits_dir: str | Path,
) -> dict[str, Path]:
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, records in splits.items():
        path = splits_dir / f"{name}.jsonl"
        save_records_jsonl(records, path)
        paths[name] = path
    return paths


def split_stats(splits: dict[str, list[SampleRecord]]) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}
    for name, records in splits.items():
        counts = Counter(r.label for r in records)
        stats[name] = {
            "n": len(records),
            "credible": counts.get("credible", 0),
            "misinfo": counts.get("misinfo", 0),
            "with_image": sum(1 for r in records if r.image_path),
        }
    return stats
