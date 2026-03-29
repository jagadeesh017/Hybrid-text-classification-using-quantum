"""
Comprehensive unit tests for critical HQNLP functions.

Run with: pytest tests/test_hqnlp.py -v
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
import yaml

# Test imports
try:
    from hqnlp import load_config
    from hqnlp.config import AppConfig, DataConfig, ModelConfig, TrainingConfig
    from hqnlp.inference.predict import InferenceError, predict_text
    from hqnlp.models import build_model
    from hqnlp.models.encoder import DebertaEncoder, masked_mean_pool
    from hqnlp.models.quantum import CQKSANBlock, QuantumTokenEncoder
    from hqnlp.utils import resolve_device, set_seed
except ImportError as e:
    pytest.skip(f"Import error: {e}", allow_module_level=True)


class TestConfig:
    """Test configuration loading and validation."""

    def test_load_config_default(self):
        """Test loading default config."""
        config = load_config("configs/default.yaml")
        assert isinstance(config, AppConfig)
        assert config.data.dataset_name in ["imdb", "ag_news", "sms_spam"]
        assert config.model.model_type in ["baseline", "reduced", "hybrid"]

    def test_load_config_hybrid(self):
        """Test loading hybrid config."""
        config = load_config("configs/hybrid.yaml")
        assert config.model.model_type == "hybrid"
        assert config.model.use_quantum is True

    def test_load_config_nonexistent(self):
        """Test error on nonexistent config file."""
        with pytest.raises(FileNotFoundError):
            load_config("configs/nonexistent.yaml")

    def test_config_with_invalid_yaml(self):
        """Test error on invalid YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            f.flush()
            with pytest.raises(ValueError):
                load_config(f.name)
            Path(f.name).unlink()

    def test_config_dataclass_coercion(self):
        """Test that config dataclasses handle missing fields gracefully."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                "experiment": {"name": "test"},
                "data": {"dataset_name": "imdb"},
                "model": {"model_type": "hybrid"},
            }, f)
            f.flush()
            config = load_config(f.name)
            assert config.experiment.name == "test"
            assert config.data.dataset_name == "imdb"
            assert config.training.batch_size == 8  # Default value
            Path(f.name).unlink()


class TestMaskedMeanPool:
    """Test masked mean pooling function."""

    def test_masked_mean_pool_basic(self):
        """Test basic masked mean pooling."""
        batch_size, seq_len, hidden_size = 2, 10, 768
        embeddings = torch.randn(batch_size, seq_len, hidden_size)
        mask = torch.ones(batch_size, seq_len)
        
        result = masked_mean_pool(embeddings, mask)
        assert result.shape == (batch_size, hidden_size)
        assert not torch.isnan(result).any()

    def test_masked_mean_pool_with_padding(self):
        """Test masked mean pooling with padding tokens."""
        batch_size, seq_len, hidden_size = 2, 10, 768
        embeddings = torch.randn(batch_size, seq_len, hidden_size)
        mask = torch.ones(batch_size, seq_len)
        mask[:, 5:] = 0  # Mask out padding
        
        result = masked_mean_pool(embeddings, mask)
        assert result.shape == (batch_size, hidden_size)
        assert not torch.isnan(result).any()

    def test_masked_mean_pool_invalid_input(self):
        """Test error on invalid input."""
        with pytest.raises(ValueError):
            masked_mean_pool(None, torch.ones(2, 10))

    def test_masked_mean_pool_batch_mismatch(self):
        """Test error on batch size mismatch."""
        embeddings = torch.randn(2, 10, 768)
        mask = torch.ones(3, 10)  # Different batch size
        with pytest.raises(ValueError):
            masked_mean_pool(embeddings, mask)


class TestQuantumEncoder:
    """Test Quantum Token Encoder."""

    def test_quantum_encoder_init(self):
        """Test initialization of quantum encoder."""
        encoder = QuantumTokenEncoder(num_qubits=4, n_layers=1)
        assert encoder.num_qubits == 4
        assert encoder.vqc is not None

    def test_quantum_encoder_invalid_qubits(self):
        """Test error on invalid qubits."""
        with pytest.raises(ValueError):
            QuantumTokenEncoder(num_qubits=0)

    def test_quantum_encoder_forward(self):
        """Test quantum encoder forward pass."""
        encoder = QuantumTokenEncoder(num_qubits=4, n_layers=1)
        batch_size, seq_len, width = 2, 10, 4
        inputs = torch.randn(batch_size, seq_len, width)
        
        outputs = encoder(inputs)
        assert outputs.shape == (batch_size, seq_len, width)
        assert not torch.isnan(outputs).any()

    def test_quantum_encoder_dimension_mismatch(self):
        """Test error on dimension mismatch."""
        encoder = QuantumTokenEncoder(num_qubits=4)
        inputs = torch.randn(2, 10, 8)  # Width != num_qubits
        
        with pytest.raises(ValueError):
            encoder(inputs)


class TestCQKSANBlock:
    """Test CQKSAN Block."""

    def test_cqksan_init(self):
        """Test CQKSAN initialization."""
        block = CQKSANBlock(
            input_dim=768,
            hidden_dim=128,
            reduced_dim=4,
            dropout=0.2,
            quantum_layers=1
        )
        assert block.projector is not None
        assert block.quantum_encoder is not None

    def test_cqksan_invalid_dims(self):
        """Test error on invalid dimensions."""
        with pytest.raises(ValueError):
            CQKSANBlock(input_dim=0, hidden_dim=128, reduced_dim=4, dropout=0.2, quantum_layers=1)

    def test_cqksan_invalid_dropout(self):
        """Test error on invalid dropout."""
        with pytest.raises(ValueError):
            CQKSANBlock(input_dim=768, hidden_dim=128, reduced_dim=4, dropout=1.5, quantum_layers=1)

    def test_cqksan_forward(self):
        """Test CQKSAN forward pass."""
        block = CQKSANBlock(
            input_dim=768,
            hidden_dim=128,
            reduced_dim=4,
            dropout=0.2,
            quantum_layers=1
        )
        batch_size, seq_len = 2, 10
        embeddings = torch.randn(batch_size, seq_len, 768)
        mask = torch.ones(batch_size, seq_len)
        
        fused, attention = block(embeddings, mask)
        assert fused.shape == (batch_size, seq_len, 768)
        assert attention.shape == (batch_size, seq_len, seq_len)
        assert not torch.isnan(fused).any()

    def test_cqksan_with_padding(self):
        """Test CQKSAN with padding."""
        block = CQKSANBlock(
            input_dim=768,
            hidden_dim=128,
            reduced_dim=4,
            dropout=0.2,
            quantum_layers=1
        )
        batch_size, seq_len = 2, 10
        embeddings = torch.randn(batch_size, seq_len, 768)
        mask = torch.ones(batch_size, seq_len)
        mask[:, 5:] = 0  # Mask padding
        
        fused, attention = block(embeddings, mask)
        # Check padding positions have near-zero attention
        assert (attention[:, :, 5:].max() < 0.1).item()


class TestModelFactory:
    """Test model factory."""

    def test_build_baseline_model(self):
        """Test building baseline model."""
        config = ModelConfig(
            model_type="baseline",
            backbone_name="microsoft/deberta-v3-small",
            num_labels=2
        )
        model = build_model(config)
        assert model is not None
        assert hasattr(model, 'forward')

    def test_build_reduced_model(self):
        """Test building reduced model."""
        config = ModelConfig(
            model_type="reduced",
            backbone_name="microsoft/deberta-v3-small",
            num_labels=2
        )
        model = build_model(config)
        assert model is not None

    def test_build_hybrid_model(self):
        """Test building hybrid model."""
        config = ModelConfig(
            model_type="hybrid",
            backbone_name="microsoft/deberta-v3-small",
            num_labels=2,
            reduced_dim=4
        )
        model = build_model(config)
        assert model is not None

    def test_build_invalid_model_type(self):
        """Test error on invalid model type."""
        config = ModelConfig(
            model_type="unknown",
            backbone_name="microsoft/deberta-v3-small",
            num_labels=2
        )
        with pytest.raises(ValueError):
            build_model(config)


class TestUtils:
    """Test utility functions."""

    def test_resolve_device(self):
        """Test device resolution."""
        device = resolve_device()
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda"]

    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        tensor1 = torch.randn(10)
        
        set_seed(42)
        tensor2 = torch.randn(10)
        
        assert torch.allclose(tensor1, tensor2)

    def test_set_seed_reproducibility(self):
        """Test reproducibility with seed."""
        set_seed(123)
        values1 = [torch.randn(1).item() for _ in range(5)]
        
        set_seed(123)
        values2 = [torch.randn(1).item() for _ in range(5)]
        
        assert values1 == values2


class TestInferenceErrors:
    """Test inference error handling."""

    def test_inference_error_with_invalid_text(self):
        """Test inference error on invalid text."""
        config = load_config("configs/default.yaml")
        checkpoint = "nonexistent.pt"
        
        with pytest.raises(InferenceError):
            predict_text("", config, checkpoint)

    def test_inference_error_with_nonexistent_checkpoint(self):
        """Test inference error on missing checkpoint."""
        config = load_config("configs/default.yaml")
        checkpoint = "nonexistent_checkpoint.pt"
        
        with pytest.raises(InferenceError):
            predict_text("This is a test", config, checkpoint)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPU:
    """Test GPU-specific functionality."""

    def test_cuda_availability(self):
        """Test CUDA is available."""
        assert torch.cuda.is_available()

    def test_tensor_on_cuda(self):
        """Test moving tensor to CUDA."""
        tensor = torch.randn(10, 10)
        tensor_cuda = tensor.to('cuda')
        assert tensor_cuda.device.type == 'cuda'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
