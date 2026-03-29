from __future__ import annotations

import logging
from typing import Tuple

from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from hqnlp.config import DataConfig, ModelConfig, TrainingConfig

logger = logging.getLogger(__name__)


DATASET_REGISTRY = {
    "imdb": {"path": "imdb", "text_column": "text", "label_column": "label", "num_labels": 2},
    "ag_news": {"path": "ag_news", "text_column": "text", "label_column": "label", "num_labels": 4},
    "sms_spam": {"path": "sms_spam", "text_column": "sms", "label_column": "label", "num_labels": 2},
}


def resolve_label_names(dataset_name: str) -> list[str]:
    if dataset_name == "imdb":
        return ["negative", "positive"]
    if dataset_name == "ag_news":
        return ["world", "sports", "business", "sci_tech"]
    if dataset_name == "sms_spam":
        return ["ham", "spam"]
    return []


def _prepare_dataset_splits(dataset_name: str, data_config: DataConfig) -> Tuple[DatasetDict, str, str, int]:
    """Prepare dataset with comprehensive error handling.
    
    Args:
        dataset_name: Name of dataset to load
        data_config: Data configuration
        
    Returns:
        tuple: (dataset, text_column, label_column, num_labels)
        
    Raises:
        ValueError: If dataset is unsupported or has invalid structure
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported: {sorted(DATASET_REGISTRY.keys())}")

    metadata = DATASET_REGISTRY[dataset_name]
    try:
        dataset = load_dataset(metadata["path"], cache_dir=data_config.cache_dir)
        logger.info(f"Loaded dataset: {dataset_name}")
    except Exception as e:
        logger.error(f"Failed to load dataset '{dataset_name}': {e}")
        raise ValueError(f"Failed to load dataset '{dataset_name}': {str(e)}") from e
        
    text_column = data_config.text_column or metadata["text_column"]
    label_column = data_config.label_column or metadata["label_column"]
    num_labels = metadata["num_labels"]

    if "train" not in dataset:
        raise ValueError(f"Dataset '{dataset_name}' does not expose a train split.")

    if "test" not in dataset:
        logger.warning(f"No test split found, creating from train split")
        split = dataset["train"].train_test_split(
            test_size=max(data_config.validation_split, 0.1),
            seed=42,
        )
        dataset = DatasetDict(train=split["train"], test=split["test"])

    train_split = dataset["train"]
    test_split = dataset["test"]
    
    logger.info(f"Train set size: {len(train_split)}, Test set size: {len(test_split)}")

    if data_config.train_samples:
        if data_config.train_samples > len(train_split):
            logger.warning(f"Requested {data_config.train_samples} samples, but only {len(train_split)} available")
        train_split = train_split.shuffle(seed=42).select(range(min(data_config.train_samples, len(train_split))))
        logger.info(f"Using {len(train_split)} training samples")
        
    if data_config.eval_samples:
        if data_config.eval_samples > len(test_split):
            logger.warning(f"Requested {data_config.eval_samples} eval samples, but only {len(test_split)} available")
        test_split = test_split.shuffle(seed=42).select(range(min(data_config.eval_samples, len(test_split))))
        logger.info(f"Using {len(test_split)} eval samples")

    return DatasetDict(train=train_split, test=test_split), text_column, label_column, num_labels


def build_dataloaders(data_config: DataConfig, model_config: ModelConfig, training_config: TrainingConfig):
    """Build data loaders with error handling.
    
    Args:
        data_config: Data configuration
        model_config: Model configuration
        training_config: Training configuration
        
    Returns:
        tuple: (train_loader, eval_loader, tokenizer, num_labels, label_names)
        
    Raises:
        ValueError: If data loading or preprocessing fails
    """
    dataset, text_column, label_column, num_labels = _prepare_dataset_splits(
        data_config.dataset_name, data_config
    )
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_config.backbone_name, use_fast=True)
        logger.info(f"Loaded tokenizer: {model_config.backbone_name}")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise ValueError(f"Failed to load tokenizer '{model_config.backbone_name}': {str(e)}") from e

    def preprocess(batch):
        try:
            encoded = tokenizer(
                batch[text_column],
                truncation=True,
                max_length=data_config.max_length,
            )
            encoded["labels"] = batch[label_column]
            return encoded
        except Exception as e:
            logger.error(f"Failed to preprocess batch: {e}")
            raise

    try:
        tokenized = dataset.map(
            preprocess,
            batched=True,
            desc=f"Tokenizing {data_config.dataset_name}",
        )
    except Exception as e:
        logger.error(f"Failed to tokenize dataset: {e}")
        raise ValueError(f"Dataset preprocessing failed: {str(e)}") from e
        
    columns_to_remove = [
        column
        for column in [text_column, label_column]
        if column in tokenized["train"].column_names
    ]
    tokenized = tokenized.remove_columns(columns_to_remove)
    tokenized.set_format(type="torch")

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    
    try:
        train_loader = DataLoader(
            tokenized["train"],
            batch_size=training_config.batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=data_config.num_workers,
            pin_memory=data_config.pin_memory,
        )
        eval_loader = DataLoader(
            tokenized["test"],
            batch_size=training_config.eval_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=data_config.num_workers,
            pin_memory=data_config.pin_memory,
        )
        logger.info(f"Created data loaders: train={len(train_loader)}, eval={len(eval_loader)}")
    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        raise ValueError(f"Failed to create data loaders: {str(e)}") from e
        
    return train_loader, eval_loader, tokenizer, num_labels, resolve_label_names(data_config.dataset_name)
