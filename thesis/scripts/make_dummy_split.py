#!/usr/bin/env python3
"""Create a tiny dummy split so folder layout and loaders can be tested before real data."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.schema import SampleRecord
from src.data.split import save_splits, split_stats, stratified_split


def main() -> None:
    records = [
        SampleRecord(
            id=f"dummy-{i}",
            text=text,
            label=label,
            image_path=None,
            platform="x",
            source="dummy",
        )
        for i, (text, label) in enumerate(
            [
                ("Vaccines reduce hospitalization risk.", "credible"),
                ("Vaccines contain tracking microchips.", "misinfo"),
                ("Booster shots help against new variants.", "credible"),
                ("mRNA vaccines permanently alter DNA.", "misinfo"),
                ("Public health agencies monitor vaccine safety.", "credible"),
                ("COVID vaccines make people magnetic.", "misinfo"),
                ("Immunization campaigns protect communities.", "credible"),
                ("Vaccine side effects are always fatal.", "misinfo"),
            ],
            start=1,
        )
    ]
    splits = stratified_split(records, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=42)
    paths = save_splits(splits, ROOT / "data" / "splits")
    print(split_stats(splits))
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
