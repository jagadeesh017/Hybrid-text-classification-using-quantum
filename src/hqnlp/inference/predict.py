from __future__ import annotations

import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer

from hqnlp.config import AppConfig
from hqnlp.models import build_model
from hqnlp.utils import resolve_device

logger = logging.getLogger(__name__)


class InferenceError(Exception):
    """Custom exception for inference-related errors."""
    pass


def load_model_for_inference(config: AppConfig, checkpoint_path: str):
    """Load model and tokenizer with comprehensive error handling.
    
    Args:
        config: AppConfig instance
        checkpoint_path: Path to model checkpoint
        
    Returns:
        tuple: (model, tokenizer, device)
        
    Raises:
        InferenceError: If model loading fails
    """
    try:
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            raise InferenceError(f"Checkpoint not found: {checkpoint_path}")
        if not checkpoint_file.is_file():
            raise InferenceError(f"Checkpoint path is not a file: {checkpoint_path}")
            
        device = resolve_device()
        logger.info(f"Using device: {device}")
        
        model = build_model(config.model)
        logger.info(f"Built model: {config.model.model_type}")
        
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
        
        tokenizer_dir = Path(checkpoint_path).parent / "tokenizer"
        if tokenizer_dir.exists():
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
            logger.info(f"Loaded tokenizer from: {tokenizer_dir}")
        else:
            tokenizer = AutoTokenizer.from_pretrained(config.model.backbone_name, use_fast=True)
            logger.info(f"Loaded tokenizer from: {config.model.backbone_name}")
            
        return model, tokenizer, device
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise InferenceError(f"Model loading failed: {str(e)}") from e


@torch.no_grad()
def predict_text(text: str, config: AppConfig, checkpoint_path: str, class_names: list[str] | None = None) -> dict:
    """Predict class for input text.
    
    Args:
        text: Input text to classify
        config: AppConfig instance
        checkpoint_path: Path to model checkpoint
        class_names: Optional list of class names
        
    Returns:
        dict: Prediction result with label, confidence, and probabilities
        
    Raises:
        InferenceError: If prediction fails
    """
    if not text or not isinstance(text, str):
        raise InferenceError("Input text must be a non-empty string")
    if len(text.strip()) == 0:
        raise InferenceError("Input text cannot be empty or whitespace-only")
        
    try:
        model, tokenizer, device = load_model_for_inference(config, checkpoint_path)
        
        encoded = tokenizer(text, truncation=True, max_length=config.data.max_length, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        
        outputs = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
        probabilities = torch.softmax(outputs["logits"], dim=-1)[0]
        pred = int(torch.argmax(probabilities).item())
        
        labels = class_names or config.inference.class_names or [str(i) for i in range(config.model.num_labels)]
        
        if pred >= len(labels):
            logger.warning(f"Prediction {pred} exceeds label count {len(labels)}")
            
        return {
            "label_id": pred,
            "label": labels[pred] if pred < len(labels) else str(pred),
            "confidence": float(probabilities[pred].item()),
            "probabilities": probabilities.cpu().tolist(),
        }
    except InferenceError:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise InferenceError(f"Prediction failed: {str(e)}") from e
