from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

# Canonical labels for the survival-scope binary setup.
LABEL_MAP: dict[str, int] = {
    "credible": 0,
    "misinfo": 1,
}

ID_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_MAP.items()}

# Common aliases found in public COVID/vaccine misinfo datasets.
_LABEL_ALIASES: dict[str, str] = {
    "credible": "credible",
    "real": "credible",
    "true": "credible",
    "reliable": "credible",
    "0": "credible",
    "misinfo": "misinfo",
    "misinformation": "misinfo",
    "fake": "misinfo",
    "false": "misinfo",
    "misleading": "misinfo",
    "unreliable": "misinfo",
    "1": "misinfo",
}


def normalize_label(raw: str | int) -> str:
    """Map heterogeneous dataset labels to {credible, misinfo}."""
    key = str(raw).strip().lower()
    if key not in _LABEL_ALIASES:
        raise ValueError(f"Unsupported label: {raw!r}. Expected aliases of credible/misinfo.")
    return _LABEL_ALIASES[key]


@dataclass
class SampleRecord:
    """One multimodal sample after preprocessing."""

    id: str
    text: str
    label: str
    image_path: str | None = None
    platform: str | None = None  # x | tiktok | other
    source: str | None = None
    split: str | None = None  # train | val | test

    def label_id(self) -> int:
        return LABEL_MAP[normalize_label(self.label)]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
