#!/usr/bin/env python3
"""
Environment Setup Script for Hybrid Quantum-Classical NLP Model

This script prepares the environment for running the inference pipeline
or training models. It verifies dependencies and creates required directories.

Usage:
    python setup.py

Requirements:
    - Python 3.8+
    - pip with --upgrade capability
    - See requirements.txt for package dependencies
"""

import os
import sys
import shutil
from pathlib import Path

def main():
    """Initialize environment and verify setup"""
    print("=" * 70)
    print("Hybrid Quantum-Classical NLP - Environment Setup")
    print("=" * 70)
    
    success = True
    
    # Step 1: Create required directories
    print("\n[1/3] Creating required directories...")
    model_dir = Path("artifacts/runs/hybrid_cqksan_deberta_imdb")
    try:
        model_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = Path("artifacts/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"      ✓ Directory structure created")
    except Exception as e:
        print(f"      ✗ Failed to create directories: {e}")
        success = False
    
    # Step 2: Verify Python dependencies
    print("\n[2/3] Verifying Python dependencies...")
    required_packages = [
        'torch', 'pennylane', 'gradio', 'transformers', 'datasets'
    ]
    missing = []
    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"      ✓ {pkg}")
        except ImportError:
            print(f"      ✗ {pkg} (missing)")
            missing.append(pkg)
    
    if missing:
        print(f"\n      Run: pip install -r requirements.txt")
        success = False
    
    # Step 3: Check model availability
    print("\n[3/3] Checking pre-trained model availability...")
    model_path = Path("artifacts/runs/hybrid_cqksan_deberta_imdb/best_model.pt")
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"      ✓ Model checkpoint found ({size_mb:.1f} MB)")
    else:
        print(f"      ⚠ Pre-trained model not available")
        print(f"      → Train model: python train.py --config configs/hybrid.yaml")
        print(f"      → Or request pre-trained checkpoint from maintainer")
        success = False
    
    # Final status
    print("\n" + "=" * 70)
    if success and model_path.exists():
        print("✅ Environment Ready - Launch app with:")
        print("   python app.py --config configs/hybrid.yaml")
    elif success:
        print("⚠ Environment Ready (Missing Model) - Train or import checkpoint first")
    else:
        print("❌ Setup incomplete - Install dependencies and try again")
    print("=" * 70)

if __name__ == "__main__":
    main()
