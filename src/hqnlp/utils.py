from __future__ import annotations

import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_serializable(payload: Any) -> Any:
    if is_dataclass(payload):
        return {key: to_serializable(value) for key, value in asdict(payload).items()}
    if isinstance(payload, dict):
        return {key: to_serializable(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [to_serializable(item) for item in payload]
    return payload


def save_json(payload: Any, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_serializable(payload), handle, indent=2)


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
