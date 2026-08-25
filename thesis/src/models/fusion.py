from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

from .image_encoder import ClipImageEncoder


class LateFusionClassifier(nn.Module):
    """Late fusion: RoBERTa CLS ⊕ CLIP image embedding → MLP."""

    def __init__(
        self,
        text_model_name: str = "roberta-base",
        image_model_name: str = "openai/clip-vit-base-patch32",
        num_labels: int = 2,
        dropout: float = 0.2,
        freeze_image_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.image_encoder = ClipImageEncoder(
            model_name=image_model_name,
            freeze=freeze_image_encoder,
        )
        text_hidden = int(self.text_encoder.config.hidden_size)
        image_dim = int(self.image_encoder.out_dim)
        fused_dim = text_hidden + image_dim
        self.dropout = nn.Dropout(dropout)
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **_: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_cls = text_out.last_hidden_state[:, 0]

        if pixel_values is None:
            image_embeds = torch.zeros(
                text_cls.size(0),
                self.image_encoder.out_dim,
                device=text_cls.device,
                dtype=text_cls.dtype,
            )
        else:
            image_embeds = self.image_encoder(pixel_values)

        fused = torch.cat([self.dropout(text_cls), self.dropout(image_embeds)], dim=-1)
        logits = self.fusion_head(fused)
        result: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            result["loss"] = nn.functional.cross_entropy(logits, labels)
        return result
