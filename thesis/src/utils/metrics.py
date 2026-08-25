from __future__ import annotations

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: list[int] | Any,
    y_pred: list[int] | Any,
    average: str = "macro",
) -> dict[str, float]:
    """Compute accuracy / precision / recall / F1 for classification."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }


def format_classification_report(
    y_true: list[int] | Any,
    y_pred: list[int] | Any,
    target_names: list[str] | None = None,
) -> str:
    return classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0,
        digits=4,
    )
