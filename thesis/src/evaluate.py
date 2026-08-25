from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.engine import build_loaders, evaluate_loader, get_device
from src.models.fusion import LateFusionClassifier
from src.models.image_encoder import ImageClassifier
from src.models.text_classifier import TextClassifier
from src.utils.config import load_config
from src.utils.metrics import format_classification_report
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate vaccine misinfo classifier")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/text_best.pt")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--modality",
        type=str,
        default=None,
        choices=["text", "image", "fusion"],
        help="Override modality (default: read from checkpoint)",
    )
    return parser.parse_args()


def build_model_from_ckpt(cfg: dict, ckpt: dict, modality: str) -> torch.nn.Module:
    text_name = ckpt.get("text_model_name") or ckpt.get("model_name") or cfg["model"]["text_encoder"]
    image_name = ckpt.get("image_model_name", cfg["model"]["image_encoder"])
    num_labels = int(ckpt.get("num_labels", cfg["model"]["num_labels"]))
    dropout = float(ckpt.get("dropout", cfg["model"]["dropout"]))
    freeze_image = bool(ckpt.get("freeze_image_encoder", cfg["model"].get("freeze_image_encoder", True)))

    if modality == "text":
        return TextClassifier(model_name=text_name, num_labels=num_labels, dropout=dropout)
    if modality == "image":
        return ImageClassifier(
            model_name=image_name,
            num_labels=num_labels,
            dropout=dropout,
            freeze_encoder=freeze_image,
        )
    return LateFusionClassifier(
        text_model_name=text_name,
        image_model_name=image_name,
        num_labels=num_labels,
        dropout=dropout,
        freeze_image_encoder=freeze_image,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["project"]["seed"]))

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = get_device()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    modality = args.modality or ckpt.get("modality") or cfg["model"]["modality"]
    text_name = ckpt.get("text_model_name") or ckpt.get("model_name") or cfg["model"]["text_encoder"]
    image_name = ckpt.get("image_model_name", cfg["model"]["image_encoder"])

    loaders = build_loaders(
        cfg,
        modality=modality,
        text_model_name=text_name,
        image_model_name=image_name,
        root_dir=Path.cwd(),
    )
    model = build_model_from_ckpt(cfg, ckpt, modality).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    metrics = evaluate_loader(
        model,
        loaders[args.split],
        device,
        average=cfg["eval"]["average"],
    )

    # Detailed report
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loaders[args.split]:
            labels = batch["labels"].to(device)
            kwargs = {"labels": labels}
            if "input_ids" in batch:
                kwargs["input_ids"] = batch["input_ids"].to(device)
                kwargs["attention_mask"] = batch["attention_mask"].to(device)
            if "pixel_values" in batch:
                kwargs["pixel_values"] = batch["pixel_values"].to(device)
            out = model(**kwargs)
            y_true.extend(batch["labels"].tolist())
            y_pred.extend(out["logits"].argmax(dim=-1).cpu().tolist())
    report = format_classification_report(y_true, y_pred, target_names=["credible", "misinfo"])

    print(f"[evaluate] split={args.split} modality={modality} checkpoint={ckpt_path} device={device}")
    print(json.dumps(metrics, indent=2))
    print(report)

    out_dir = Path(cfg["project"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval_{modality}_{args.split}.json"
    out_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "modality": modality,
                "split": args.split,
                "checkpoint": str(ckpt_path),
                "metrics": metrics,
                "report": report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[evaluate] wrote {out_path}")


if __name__ == "__main__":
    main()
