from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from .preprocess import load_records_jsonl
from .schema import SampleRecord

# Allow partially downloaded / truncated JPEGs when possible
ImageFile.LOAD_TRUNCATED_IMAGES = True


class VaccineMisinfoDataset(Dataset):
    """PyTorch dataset for text (+ optional image) vaccine misinfo samples."""

    def __init__(
        self,
        records: list[SampleRecord] | str | Path,
        *,
        tokenizer: Any | None = None,
        image_processor: Any | None = None,
        image_transform: Callable[[Image.Image], Any] | None = None,
        max_length: int = 256,
        require_image: bool = False,
        root_dir: str | Path | None = None,
    ) -> None:
        if isinstance(records, (str, Path)):
            self.records = load_records_jsonl(records)
        else:
            self.records = records

        if require_image:
            self.records = [r for r in self.records if r.image_path]

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_transform = image_transform
        self.max_length = max_length
        self.root_dir = Path(root_dir) if root_dir else Path.cwd()

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_image_path(self, image_path: str | None) -> Path | None:
        if not image_path:
            return None
        path = Path(image_path)
        if not path.is_absolute():
            path = self.root_dir / path
        return path if path.exists() else None

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.records[idx]
        item: dict[str, Any] = {
            "id": sample.id,
            "text": sample.text,
            "label": sample.label_id(),
        }

        if self.tokenizer is not None:
            encoded = self.tokenizer(
                sample.text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            item["input_ids"] = encoded["input_ids"].squeeze(0)
            item["attention_mask"] = encoded["attention_mask"].squeeze(0)

        img_path = self._resolve_image_path(sample.image_path)
        if self.image_processor is not None:
            pixel_values = torch.zeros(3, 224, 224)
            has_image = False
            if img_path is not None:
                try:
                    image = Image.open(img_path).convert("RGB")
                    proc = self.image_processor(images=image, return_tensors="pt")
                    pixel_values = proc["pixel_values"].squeeze(0)
                    has_image = True
                except (OSError, ValueError, Image.DecompressionBombError) as e:
                    # Corrupt/truncated downloads — keep zero placeholder
                    print(f"[warn] bad image {img_path}: {e}")
            item["pixel_values"] = pixel_values
            item["has_image"] = has_image
        elif self.image_transform is not None:
            if img_path is not None:
                try:
                    image = Image.open(img_path).convert("RGB")
                    item["pixel_values"] = self.image_transform(image)
                    item["has_image"] = True
                except (OSError, ValueError):
                    item["pixel_values"] = None
                    item["has_image"] = False
            else:
                item["pixel_values"] = None
                item["has_image"] = False
        else:
            item["has_image"] = img_path is not None

        return item
