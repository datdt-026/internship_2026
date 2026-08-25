from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.engine import build_loaders, evaluate_loader, get_device, train_one_epoch
from src.models.fusion import LateFusionClassifier
from src.models.image_encoder import ImageClassifier
from src.models.text_classifier import TextClassifier
from src.utils.config import load_config
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train vaccine misinfo classifier")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--modality",
        type=str,
        choices=["text", "image", "fusion"],
        default=None,
        help="Override model.modality from config",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def build_model(cfg: dict, modality: str) -> torch.nn.Module:
    text_name = cfg["model"]["text_encoder"]
    image_name = cfg["model"]["image_encoder"]
    num_labels = int(cfg["model"]["num_labels"])
    dropout = float(cfg["model"]["dropout"])
    freeze_image = bool(cfg["model"].get("freeze_image_encoder", True))

    if modality == "text":
        return TextClassifier(model_name=text_name, num_labels=num_labels, dropout=dropout)
    if modality == "image":
        return ImageClassifier(
            model_name=image_name,
            num_labels=num_labels,
            dropout=dropout,
            freeze_encoder=freeze_image,
        )
    if modality == "fusion":
        return LateFusionClassifier(
            text_model_name=text_name,
            image_model_name=image_name,
            num_labels=num_labels,
            dropout=dropout,
            freeze_image_encoder=freeze_image,
        )
    raise ValueError(f"Unknown modality: {modality}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.modality:
        cfg["model"]["modality"] = args.modality
    if args.epochs is not None:
        cfg["train"]["num_epochs"] = args.epochs

    set_seed(int(cfg["project"]["seed"]))
    modality = cfg["model"]["modality"]
    device = get_device()
    text_name = cfg["model"]["text_encoder"]
    image_name = cfg["model"]["image_encoder"]

    print(f"[train] project={cfg['project']['name']} modality={modality} device={device}")
    print(f"[train] splits_dir={cfg['data']['splits_dir']}")
    print(f"[train] text={text_name} image={image_name}")

    loaders = build_loaders(
        cfg,
        modality=modality,
        text_model_name=text_name,
        image_model_name=image_name,
        root_dir=Path.cwd(),
    )
    model = build_model(cfg, modality).to(device)

    # Only optimize trainable params (frozen CLIP stays out)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    out_dir = Path(cfg["project"]["output_dir"])
    ckpt_dir = Path(cfg["project"]["checkpoint_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    patience = int(cfg["train"]["early_stopping_patience"])
    bad_epochs = 0
    history: list[dict] = []
    best_path = ckpt_dir / f"{modality}_best.pt"

    num_epochs = int(cfg["train"]["num_epochs"])
    for epoch in range(1, num_epochs + 1):
        print(f"\n[train] epoch {epoch}/{num_epochs}")
        train_metrics = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            device,
            max_steps=args.max_steps,
            logging_steps=int(cfg["train"]["logging_steps"]),
        )
        val_metrics = evaluate_loader(
            model,
            loaders["val"],
            device,
            average=cfg["eval"]["average"],
        )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"[train] epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['f1']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "modality": modality,
                    "text_model_name": text_name,
                    "image_model_name": image_name,
                    "num_labels": int(cfg["model"]["num_labels"]),
                    "dropout": float(cfg["model"]["dropout"]),
                    "freeze_image_encoder": bool(cfg["model"].get("freeze_image_encoder", True)),
                    "label_map": cfg["data"]["label_map"],
                    "val_metrics": val_metrics,
                    "epoch": epoch,
                    # legacy keys for Step-3 evaluate compatibility
                    "model_name": text_name,
                },
                best_path,
            )
            print(f"[train] saved best checkpoint -> {best_path} (val_f1={best_f1:.4f})")
        else:
            bad_epochs += 1
            print(f"[train] no val_f1 improvement ({bad_epochs}/{patience})")
            if bad_epochs >= patience:
                print("[train] early stopping")
                break

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate_loader(
        model,
        loaders["test"],
        device,
        average=cfg["eval"]["average"],
    )
    print(
        f"\n[train] TEST accuracy={test_metrics['accuracy']:.4f} "
        f"precision={test_metrics['precision']:.4f} "
        f"recall={test_metrics['recall']:.4f} "
        f"f1={test_metrics['f1']:.4f}"
    )

    result = {
        "status": "ok",
        "modality": modality,
        "splits_dir": cfg["data"]["splits_dir"],
        "text_model_name": text_name,
        "image_model_name": image_name,
        "device": str(device),
        "best_val_f1": best_f1,
        "best_checkpoint": str(best_path),
        "test_metrics": test_metrics,
        "history": history,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    result_path = out_dir / f"{modality}_results.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    # Keep Step-3 friendly alias for text modality
    if modality == "text" and "constraint" in str(cfg["data"]["splits_dir"]):
        alias = out_dir / "text_baseline_results.json"
        alias.write_text(json.dumps(result, indent=2), encoding="utf-8")
        legacy_ckpt = ckpt_dir / "text_roberta_best.pt"
        if best_path.exists() and best_path.resolve() != legacy_ckpt.resolve():
            import shutil

            shutil.copy2(best_path, legacy_ckpt)
    print(f"[train] wrote {result_path}")


if __name__ == "__main__":
    main()
