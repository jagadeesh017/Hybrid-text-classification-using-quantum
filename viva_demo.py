#!/usr/bin/env python
"""
Quick viva demonstration script - shows system working in < 1 minute.

Run this during viva to demonstrate:
1. Configuration loading
2. Model building  
3. Text classification
4. Results visualization

Usage:
    python viva_demo.py
"""

import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

def demo_configuration():
    """Demo 1: Configuration Loading"""
    print("\n" + "="*70)
    print("DEMO 1: Configuration System")
    print("="*70)
    
    from hqnlp import load_config
    
    print("\nLoading hybrid configuration...")
    config = load_config("configs/hybrid.yaml")
    
    print(f"✓ Model Type: {config.model.model_type}")
    print(f"✓ Backbone: {config.model.backbone_name}")
    print(f"✓ Dataset: {config.data.dataset_name}")
    print(f"✓ Training Epochs: {config.training.epochs}")
    print(f"✓ Quantum Layers: {config.model.quantum_layers}")
    print(f"✓ Reduced Dimension (Qubits): {config.model.reduced_dim}")


def demo_model_building():
    """Demo 2: Model Architecture"""
    print("\n" + "="*70)
    print("DEMO 2: Model Architecture")
    print("="*70)
    
    import torch
    from hqnlp import load_config, build_model
    
    config = load_config("configs/hybrid.yaml")
    
    print(f"\nBuilding {config.model.model_type.upper()} model...")
    model = build_model(config.model)
    
    print(f"✓ Model built successfully")
    print(f"\nModel Architecture:")
    print(f"  Input: Text tokens (batch, seq_len)")
    print(f"  ↓ DeBERTa Encoder")
    print(f"  ↓ Dual paths:")
    print(f"    • Classical: Token pooling → 768-dim")
    print(f"    • Quantum: Projection → {config.model.reduced_dim} qubits")
    print(f"             → Quantum circuit")
    print(f"             → Kernel attention")
    print(f"  ↓ Fusion: Concatenate + Linear")
    print(f"  Output: Logits ({config.model.num_labels} classes)")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total Parameters: {total_params:,}")
    print(f"  Trainable: {trainable:,}")


def demo_inference():
    """Demo 3: Text Classification"""
    print("\n" + "="*70)
    print("DEMO 3: Text Classification Inference")
    print("="*70)
    
    from hqnlp import load_config
    from hqnlp.data import resolve_label_names
    from hqnlp.inference.predict import predict_text
    
    config = load_config("configs/hybrid.yaml")
    
    # Check if checkpoint exists
    checkpoint_path = Path("artifacts/runs/hybrid_cqksan_deberta_imdb/best_model.pt")
    
    if not checkpoint_path.exists():
        print(f"\n⚠️  Model checkpoint not found at: {checkpoint_path}")
        print("\nTo train a model, run:")
        print("  python train.py --config configs/hybrid.yaml")
        print("\nOr run all experiments:")
        print("  python run_experiments.py --all")
        return
    
    print(f"\n✓ Checkpoint found: {checkpoint_path}")
    print("\nRunning inference on sample texts...\n")
    
    label_names = resolve_label_names(config.data.dataset_name)
    
    # Demo texts
    texts = [
        "This movie was absolutely fantastic! Great performances and storyline.",
        "Terrible waste of time. Poor acting and boring plot.",
        "It was okay, nothing special but watchable.",
    ]
    
    try:
        for i, text in enumerate(texts, 1):
            print(f"{i}. Input: \"{text}\"")
            
            result = predict_text(text, config, str(checkpoint_path), label_names)
            
            print(f"   Prediction: {result['label'].upper()}")
            print(f"   Confidence: {result['confidence']*100:.1f}%")
            print(f"   Probabilities: {[f'{p:.3f}' for p in result['probabilities']]}")
            print()
            
    except Exception as e:
        print(f"\n⚠️  Inference error: {e}")
        print("\nThis is expected if checkpoint hasn't been trained yet.")


def demo_results():
    """Demo 4: Experimental Results"""
    print("\n" + "="*70)
    print("DEMO 4: Experimental Results")
    print("="*70)
    
    results_file = Path("results/RESULTS.md")
    
    if not results_file.exists():
        print(f"\n⚠️  Results file not found: {results_file}")
        print("\nTo generate results, run:")
        print("  python run_experiments.py --all")
        print("\nThis will:")
        print("  1. Train baseline model")
        print("  2. Train reduced model")
        print("  3. Train hybrid model (with quantum)")
        print("  4. Compare performance")
        print("  5. Generate comparison table and analysis")
        return
    
    print(f"\n✓ Results file found: {results_file}\n")
    
    # Read and display results
    with open(results_file, 'r') as f:
        content = f.read()
        # Display only first 50 lines (summary)
        lines = content.split('\n')[:40]
        for line in lines:
            print(line)
        if len(content.split('\n')) > 40:
            print("\n... (truncated) ...\n")
            print("Full results in: results/RESULTS.md")


def demo_validation():
    """Demo 5: Project Validation"""
    print("\n" + "="*70)
    print("DEMO 5: Project Health Check")
    print("="*70)
    
    checks = {
        "Configuration files": [
            "configs/default.yaml",
            "configs/hybrid.yaml",
            "configs/baseline.yaml",
            "configs/reduced.yaml",
        ],
        "Source code": [
            "src/hqnlp/__init__.py",
            "src/hqnlp/models/quantum.py",
            "src/hqnlp/models/factory.py",
            "src/hqnlp/inference/predict.py",
            "src/hqnlp/training/trainer.py",
        ],
        "Documentation": [
            "README.md",
            "docs/PRODUCTION_SETUP.md",
            "docs/VIVA_GUIDE.md",
            "docs/DEMO_SCRIPT.md",
        ],
        "Entry points": [
            "train.py",
            "inference.py",
            "app.py",
            "validate_setup.py",
            "run_experiments.py",
        ],
    }
    
    for category, files in checks.items():
        print(f"\n{category}:")
        for filepath in files:
            path = Path(filepath)
            status = "✓" if path.exists() else "✗"
            print(f"  {status} {filepath}")


def main():
    """Run all demos."""
    print("\n" + "█"*70)
    print("CQKSAN-DeBERTa Hybrid Quantum NLP - VIVA DEMONSTRATION")
    print("█"*70)
    
    try:
        # Demo 1: Configuration
        demo_configuration()
        
        # Demo 2: Model Building
        demo_model_building()
        
        # Demo 3: Inference
        demo_inference()
        
        # Demo 4: Results
        demo_results()
        
        # Demo 5: Validation
        demo_validation()
        
        # Closing
        print("\n" + "="*70)
        print("DEMONSTRATION COMPLETE")
        print("="*70)
        print("\n📋 Next Steps for Viva:")
        print("  1. Run: python run_experiments.py --all")
        print("     (This trains all 3 models and generates results)")
        print("\n  2. Show results: cat results/RESULTS.md")
        print("     (See comparison of baseline, reduced, hybrid)")
        print("\n  3. Run inference demo: python viva_demo.py")
        print("     (This will show classifications on sample texts)")
        print("\n  4. Reference guide: docs/VIVA_GUIDE.md")
        print("     (Contains all viva questions and answers)")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
