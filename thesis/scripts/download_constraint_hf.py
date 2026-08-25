#!/usr/bin/env python3
"""Download CONSTRAINT COVID fake news dataset from Hugging Face into data/raw."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw" / "constraint_covid_fake_news"
HF_ID = "nanyy1025/covid_fake_news"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {HF_ID} ...")
    ds = load_dataset(HF_ID)
    print(ds)

    frames = []
    for split in ds.keys():
        df = ds[split].to_pandas()
        path = OUT_DIR / f"{split}.csv"
        df.to_csv(path, index=False)
        print(f"saved {split}: {df.shape} -> {path}")
        df = df.copy()
        df["hf_split"] = split
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined_path = OUT_DIR / "all.csv"
    combined.to_csv(combined_path, index=False)
    print(f"saved combined: {combined.shape} -> {combined_path}")
    if "label" in combined.columns:
        print(combined["label"].value_counts(dropna=False))
    print("DONE")


if __name__ == "__main__":
    main()
