from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    name: str = "cqksan-experiment"
    seed: int = 42
    output_dir: str = "artifacts/runs"


@dataclass
class DataConfig:
    dataset_name: str = "imdb"
    text_column: Optional[str] = None
    label_column: Optional[str] = None
    max_length: int = 128
    train_samples: Optional[int] = 2000
    eval_samples: Optional[int] = 400
    validation_split: float = 0.1
    cache_dir: str = "artifacts/cache"
    num_workers: int = 0
    pin_memory: bool = True


@dataclass
class ModelConfig:
    model_type: str = "hybrid"
    backbone_name: str = "microsoft/deberta-v3-small"
    num_labels: int = 2
    dropout: float = 0.2
    freeze_backbone: bool = True
    unfreeze_last_n: int = 0
    projector_hidden_dim: int = 128
    reduced_dim: int = 4
    quantum_layers: int = 1
    quantum_attention_heads: int = 1
    classifier_hidden_dim: int = 128
    use_quantum: bool = True


@dataclass
class TrainingConfig:
    batch_size: int = 8
    eval_batch_size: int = 8
    learning_rate: float = 5e-4
    weight_decay: float = 1e-2
    epochs: int = 3
    warmup_ratio: float = 0.1
    grad_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    early_stopping_patience: int = 2
    metric_for_best_model: str = "f1"


@dataclass
class InferenceConfig:
    checkpoint_path: str = ""
    class_names: list[str] | None = None


@dataclass
class AppConfig:
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    inference: InferenceConfig


def _coerce_dataclass(dataclass_type: type[Any], payload: dict[str, Any]) -> Any:
    valid = {field.name for field in fields(dataclass_type)}
    filtered = {key: value for key, value in payload.items() if key in valid}
    return dataclass_type(**filtered)


def load_config(path: str | Path) -> AppConfig:
    """Load configuration from YAML file with validation.
    
    Args:
        path: Path to YAML configuration file
        
    Returns:
        AppConfig: Loaded configuration object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
        ValueError: If configuration is invalid
    """
    config_path = Path(path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        logger.info(f"Loaded config from: {config_path}")
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML config: {e}")
        raise ValueError(f"Invalid YAML in config file: {str(e)}") from e
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        raise

    return AppConfig(
        experiment=_coerce_dataclass(ExperimentConfig, payload.get("experiment", {})),
        data=_coerce_dataclass(DataConfig, payload.get("data", {})),
        model=_coerce_dataclass(ModelConfig, payload.get("model", {})),
        training=_coerce_dataclass(TrainingConfig, payload.get("training", {})),
        inference=_coerce_dataclass(InferenceConfig, payload.get("inference", {})),
    )
