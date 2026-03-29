from __future__ import annotations

import logging
import math

import pennylane as qml
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class QuantumTokenEncoder(nn.Module):
    """Quantum kernel-inspired token encoder.
    
    Args:
        num_qubits: Number of qubits
        n_layers: Number of quantum layers
    """
    def __init__(self, num_qubits: int = 4, n_layers: int = 1):
        super().__init__()
        if num_qubits < 1:
            raise ValueError("num_qubits must be >= 1")
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
            
        self.num_qubits = num_qubits
        self.dev = qml.device("default.qubit", wires=num_qubits)
        logger.info(f"Initialized quantum device with {num_qubits} qubits, {n_layers} layers")

        @qml.qnode(self.dev, interface="torch")
        def qnode(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(num_qubits), rotation="Y")
            for layer in range(n_layers):
                for wire in range(num_qubits):
                    qml.RY(weights[layer, wire], wires=wire)
                for wire in range(num_qubits - 1):
                    qml.CNOT(wires=[wire, wire + 1])
            return [qml.expval(qml.PauliZ(wire)) for wire in range(num_qubits)]

        self.vqc = qml.qnn.TorchLayer(qnode, {"weights": (n_layers, num_qubits)})

    def forward(self, projected_tokens: torch.Tensor) -> torch.Tensor:
        """Process tokens through quantum circuit.
        
        Args:
            projected_tokens: Token projections (batch, seq_len, width)
            
        Returns:
            torch.Tensor: Quantum-encoded tokens (batch, seq_len, width)
        """
        if projected_tokens is None:
            raise ValueError("projected_tokens cannot be None")
            
        batch, seq_len, width = projected_tokens.shape
        if width != self.num_qubits:
            raise ValueError(f"Input width {width} doesn't match num_qubits {self.num_qubits}")
        
        flat = projected_tokens.reshape(batch * seq_len, width)
        try:
            encoded = self.vqc(flat).to(torch.float32)
        except Exception as e:
            logger.error(f"Quantum encoding failed: {e}")
            raise
            
        encoded = torch.nan_to_num(encoded, nan=0.0, posinf=1.0, neginf=-1.0)
        return encoded.reshape(batch, seq_len, width)


class CQKSANBlock(nn.Module):
    """Compact Quantum Kernel Self-Attention Network Block.
    
    Combines quantum-kernel-inspired attention with classical projections.
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        reduced_dim: Quantum reduced dimension
        dropout: Dropout rate
        quantum_layers: Number of quantum layers
    """
    def __init__(self, input_dim: int, hidden_dim: int, reduced_dim: int, dropout: float, quantum_layers: int):
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or reduced_dim < 1:
            raise ValueError("All dimensions must be positive integers")
        if not 0 <= dropout < 1:
            raise ValueError("Dropout must be in [0, 1)")
            
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, reduced_dim),
        )
        self.quantum_encoder = QuantumTokenEncoder(num_qubits=reduced_dim, n_layers=quantum_layers)
        self.out_projection = nn.Linear(reduced_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        logger.info(f"CQKSAN: {input_dim} -> {hidden_dim} -> {reduced_dim} -> {input_dim}")

    def forward(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply quantum-kernel-inspired self-attention.
        
        Args:
            token_embeddings: Token embeddings (batch, seq_len, dim)
            attention_mask: Attention mask (batch, seq_len)
            
        Returns:
            tuple: (attended_tokens, attention_weights)
        """
        if token_embeddings is None or attention_mask is None:
            raise ValueError("token_embeddings and attention_mask are required")
        if token_embeddings.size(0) != attention_mask.size(0):
            raise ValueError("Batch size mismatch")
            
        reduced = torch.tanh(self.projector(token_embeddings)) * (math.pi * 0.9)
        quantum_tokens = self.quantum_encoder(reduced)

        normalized = torch.nn.functional.normalize(quantum_tokens, dim=-1, eps=1e-8)
        kernel_scores = torch.matmul(normalized, normalized.transpose(1, 2))
        
        # Apply attention mask
        padding_mask = attention_mask.unsqueeze(1) == 0
        kernel_scores = kernel_scores.masked_fill(padding_mask, -1e4)
        
        attention_weights = torch.softmax(kernel_scores, dim=-1)
        
        # Handle NaN in attention weights
        if torch.isnan(attention_weights).any():
            logger.warning("NaN in attention weights, applying nan_to_num")
            attention_weights = torch.nan_to_num(attention_weights, 0.0)
        
        attended = torch.matmul(attention_weights, quantum_tokens)
        fused = self.out_projection(self.dropout(attended))
        
        return fused, attention_weights
