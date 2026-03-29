from __future__ import annotations

import torch
import torch.nn as nn

from hqnlp.config import ModelConfig

from .encoder import DebertaEncoder
from .quantum import CQKSANBlock


class BaselineDebertaClassifier(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.encoder = DebertaEncoder(config)
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.encoder.hidden_size, config.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden_dim, config.num_labels),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        _, pooled = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(pooled)
        return {"logits": logits, "pooled_output": pooled}


class ReducedFeatureClassifier(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.encoder = DebertaEncoder(config)
        self.reducer = nn.Sequential(
            nn.Linear(self.encoder.hidden_size, config.projector_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.projector_hidden_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.projector_hidden_dim, config.reduced_dim),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.reduced_dim, config.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden_dim, config.num_labels),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        _, pooled = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        reduced = self.reducer(pooled)
        logits = self.classifier(reduced)
        return {"logits": logits, "pooled_output": pooled, "reduced_output": reduced}


class HybridCQKSANDebertaClassifier(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.encoder = DebertaEncoder(config)
        self.cqksan = CQKSANBlock(
            input_dim=self.encoder.hidden_size,
            hidden_dim=config.projector_hidden_dim,
            reduced_dim=config.reduced_dim,
            dropout=config.dropout,
            quantum_layers=config.quantum_layers,
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.encoder.hidden_size * 2, config.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.classifier = nn.Linear(config.classifier_hidden_dim, config.num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        token_embeddings, pooled = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Ensure all tensors are in float32 to avoid dtype mismatch
        token_embeddings = token_embeddings.float()
        pooled = pooled.float()
        attended_tokens, attention_weights = self.cqksan(token_embeddings, attention_mask)
        mask = attention_mask.unsqueeze(-1)
        hybrid_pooled = (attended_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        fused = self.fusion(torch.cat([pooled, hybrid_pooled], dim=-1))
        logits = self.classifier(fused)
        return {
            "logits": logits,
            "pooled_output": pooled,
            "hybrid_output": hybrid_pooled,
            "attention_weights": attention_weights,
        }


def build_model(config: ModelConfig) -> nn.Module:
    if config.model_type == "baseline":
        return BaselineDebertaClassifier(config)
    if config.model_type == "reduced":
        return ReducedFeatureClassifier(config)
    if config.model_type == "hybrid":
        return HybridCQKSANDebertaClassifier(config)
    raise ValueError(f"Unknown model_type '{config.model_type}'.")
