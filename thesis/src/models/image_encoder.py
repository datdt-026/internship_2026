from __future__ import annotations

import torch
import torch.nn as nn
from transformers import CLIPModel


class ClipImageEncoder(nn.Module):
    """CLIP vision tower → image embedding (projection dim, typically 512)."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.freeze = freeze
        self.clip = CLIPModel.from_pretrained(model_name)
        self.out_dim = int(self.clip.config.projection_dim)
        if freeze:
            for p in self.clip.parameters():
                p.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Explicit path — stable across transformers versions
        vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
        pooled = vision_outputs.pooler_output
        if pooled is None:
            # fallback: CLS token
            pooled = vision_outputs.last_hidden_state[:, 0, :]
        feats = self.clip.visual_projection(pooled)
        return feats


class ImageClassifier(nn.Module):
    """Image-only classifier on CLIP embeddings."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        num_labels: int = 2,
        dropout: float = 0.2,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = ClipImageEncoder(model_name=model_name, freeze=freeze_encoder)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.out_dim, num_labels)

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
        **_: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        embeds = self.encoder(pixel_values)
        logits = self.classifier(self.dropout(embeds))
        out: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            out["loss"] = nn.functional.cross_entropy(logits, labels)
        return out
