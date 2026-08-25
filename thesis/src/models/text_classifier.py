from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class TextClassifier(nn.Module):
    """Simple Transformer text classifier (RoBERTa/BERT CLS + linear head)."""

    def __init__(
        self,
        model_name: str = "roberta-base",
        num_labels: int = 2,
        dropout: float = 0.2,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        **_: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]
        logits = self.classifier(self.dropout(cls))
        result: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            result["loss"] = nn.functional.cross_entropy(logits, labels)
        return result
