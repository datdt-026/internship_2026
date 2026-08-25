from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPProcessor

from src.data.dataset import VaccineMisinfoDataset
from src.utils.metrics import compute_classification_metrics


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def text_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "ids": [b["id"] for b in batch],
        "input_ids": torch.stack([b["input_ids"] for b in batch], dim=0),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch], dim=0),
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }


def multimodal_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out = {
        "ids": [b["id"] for b in batch],
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "has_image": torch.tensor([bool(b.get("has_image")) for b in batch], dtype=torch.bool),
    }
    if "input_ids" in batch[0]:
        out["input_ids"] = torch.stack([b["input_ids"] for b in batch], dim=0)
        out["attention_mask"] = torch.stack([b["attention_mask"] for b in batch], dim=0)
    if "pixel_values" in batch[0] and batch[0]["pixel_values"] is not None:
        out["pixel_values"] = torch.stack([b["pixel_values"] for b in batch], dim=0)
    return out


def build_loaders(
    cfg: dict[str, Any],
    *,
    modality: str,
    text_model_name: str,
    image_model_name: str,
    root_dir: str | Path | None = None,
) -> dict[str, DataLoader]:
    splits_dir = Path(cfg["data"]["splits_dir"])
    max_len = int(cfg["data"]["max_text_length"])
    num_workers = int(cfg["data"].get("num_workers", 0))
    if num_workers > 0 and get_device().type == "mps":
        num_workers = 0
    root_dir = Path(root_dir) if root_dir else Path.cwd()

    tokenizer = None
    image_processor = None
    require_image = modality in {"image", "fusion"}
    if modality in {"text", "fusion"}:
        tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    if modality in {"image", "fusion"}:
        image_processor = CLIPProcessor.from_pretrained(image_model_name)

    collate = multimodal_collate if modality in {"image", "fusion"} else text_collate

    loaders: dict[str, DataLoader] = {}
    for split, batch_key in [
        ("train", "batch_size"),
        ("val", "eval_batch_size"),
        ("test", "eval_batch_size"),
    ]:
        path = splits_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. For MMCoVaR run prepare_mmcovar_splits.py; "
                "for CONSTRAINT run prepare_constraint_splits.py"
            )
        ds = VaccineMisinfoDataset(
            path,
            tokenizer=tokenizer,
            image_processor=image_processor,
            max_length=max_len,
            require_image=require_image,
            root_dir=root_dir,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=int(cfg["train"][batch_key]),
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=collate,
        )
    return loaders


# Backward-compatible alias used by older evaluate code paths
def build_text_loaders(
    cfg: dict[str, Any],
    tokenizer_name: str,
) -> tuple[Any, dict[str, DataLoader]]:
    loaders = build_loaders(
        cfg,
        modality="text",
        text_model_name=tokenizer_name,
        image_model_name=cfg["model"].get("image_encoder", "openai/clip-vit-base-patch32"),
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer, loaders


def _forward_batch(model: torch.nn.Module, batch: dict[str, Any], device: torch.device):
    labels = batch["labels"].to(device)
    kwargs: dict[str, Any] = {"labels": labels}
    if "input_ids" in batch:
        kwargs["input_ids"] = batch["input_ids"].to(device)
        kwargs["attention_mask"] = batch["attention_mask"].to(device)
    if "pixel_values" in batch:
        kwargs["pixel_values"] = batch["pixel_values"].to(device)
    return model(**kwargs)


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    average: str = "macro",
) -> dict[str, float]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        out = _forward_batch(model, batch, device)
        total_loss += float(out["loss"].item())
        n_batches += 1
        preds = out["logits"].argmax(dim=-1)
        y_true.extend(batch["labels"].tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    metrics = compute_classification_metrics(y_true, y_pred, average=average)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    max_steps: int | None = None,
    logging_steps: int = 50,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    n_batches = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    pbar = tqdm(loader, desc="train", leave=False)
    for step, batch in enumerate(pbar, start=1):
        optimizer.zero_grad(set_to_none=True)
        out = _forward_batch(model, batch, device)
        loss = out["loss"]
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1
        preds = out["logits"].argmax(dim=-1)
        y_true.extend(batch["labels"].tolist())
        y_pred.extend(preds.detach().cpu().tolist())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        if logging_steps and step % logging_steps == 0:
            print(f"  step={step} loss={loss.item():.4f}")
        if max_steps is not None and step >= max_steps:
            break

    metrics = compute_classification_metrics(y_true, y_pred, average="macro")
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics
