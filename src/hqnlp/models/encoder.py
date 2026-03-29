from __future__ import annotations

import logging

import torch
import torch.nn as nn
from transformers import AutoModel

from hqnlp.config import ModelConfig

logger = logging.getLogger(__name__)


class DebertaEncoder(nn.Module):
    """DeBERTa-based encoder for text embeddings.
    
    Args:
        config: ModelConfig instance
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        try:
            self.backbone = AutoModel.from_pretrained(config.backbone_name)
            logger.info(f"Loaded backbone: {config.backbone_name}")
        except Exception as e:
            logger.error(f"Failed to load backbone {config.backbone_name}: {e}")
            raise
            
        self.hidden_size = int(self.backbone.config.hidden_size)
        self.freeze_backbone = config.freeze_backbone
        logger.info(f"Backbone hidden size: {self.hidden_size}")

        if config.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            logger.info("Froze backbone parameters")

        if not config.freeze_backbone and config.unfreeze_last_n > 0:
            encoder_layers = getattr(self.backbone, "encoder", None)
            if encoder_layers is not None and hasattr(encoder_layers, "layer"):
                for layer in encoder_layers.layer[:-config.unfreeze_last_n]:
                    for parameter in layer.parameters():
                        parameter.requires_grad = False
                logger.info(f"Unfroze last {config.unfreeze_last_n} layers")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate embeddings for input tokens.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            
        Returns:
            tuple: (token_embeddings, pooled_output)
        """
        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask are required")
            
        context = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        with context:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            token_embeddings = outputs.last_hidden_state
            
        pooled = masked_mean_pool(token_embeddings, attention_mask)
        return token_embeddings, pooled


def masked_mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Compute masked mean pooling over sequence.
    
    Args:
        token_embeddings: Token embeddings (batch_size, seq_len, hidden_size)
        attention_mask: Attention mask (batch_size, seq_len)
        
    Returns:
        torch.Tensor: Pooled embeddings (batch_size, hidden_size)
    """
    if token_embeddings is None or attention_mask is None:
        raise ValueError("token_embeddings and attention_mask are required")
    if token_embeddings.size(0) != attention_mask.size(0):
        raise ValueError("Batch size mismatch between embeddings and mask")
        
    mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
    weighted = token_embeddings * mask
    denom = mask.sum(dim=1).clamp(min=1e-9)
    pooled = weighted.sum(dim=1) / denom
    
    # Handle NaN values
    if torch.isnan(pooled).any():
        logger.warning("NaN detected in pooled output, applying nan_to_num")
        pooled = torch.nan_to_num(pooled, 0.0)
    
    return pooled
