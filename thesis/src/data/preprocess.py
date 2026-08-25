from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import SampleRecord, normalize_label


_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Light text cleaning suitable for Transformer encoders."""
    text = text or ""
    text = _URL_RE.sub(" ", text)
    text = text.replace("\n", " ").replace("\t", " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def records_from_dataframe(
    df: pd.DataFrame,
    *,
    id_col: str = "id",
    text_col: str = "text",
    label_col: str = "label",
    image_col: str = "image_path",
    platform_col: str = "platform",
    source_col: str = "source",
) -> list[SampleRecord]:
    """Convert a raw dataframe into typed SampleRecord objects."""
    records: list[SampleRecord] = []
    for idx, row in df.iterrows():
        sample_id = str(row[id_col]) if id_col in df.columns and pd.notna(row.get(id_col)) else str(idx)
        text = clean_text(str(row[text_col])) if pd.notna(row.get(text_col)) else ""
        if not text:
            continue
        label = normalize_label(row[label_col])
        image_path = None
        if image_col in df.columns and pd.notna(row.get(image_col)):
            image_path = str(row[image_col])
        platform = str(row[platform_col]) if platform_col in df.columns and pd.notna(row.get(platform_col)) else None
        source = str(row[source_col]) if source_col in df.columns and pd.notna(row.get(source_col)) else None
        records.append(
            SampleRecord(
                id=sample_id,
                text=text,
                label=label,
                image_path=image_path,
                platform=platform,
                source=source,
            )
        )
    return records


def save_records_jsonl(records: list[SampleRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def load_records_jsonl(path: str | Path) -> list[SampleRecord]:
    path = Path(path)
    records: list[SampleRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj: dict[str, Any] = json.loads(line)
            records.append(SampleRecord(**obj))
    return records
