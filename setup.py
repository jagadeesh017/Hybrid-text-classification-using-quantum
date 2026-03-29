#!/usr/bin/env python3
"""
Automatic Setup Script for Collaborators
Run this AFTER cloning the repo to set everything up automatically
"""

import os
import sys
import shutil
from pathlib import Path

def create_folders():
    """Create required folder structure"""
    model_dir = Path("artifacts/runs/hybrid_cqksan_deberta_imdb")
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created folder: {model_dir}")

def install_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing dependencies...")
    result = os.system("pip install -r requirements.txt -q")
    if result == 0:
        print("✓ Dependencies installed")
    else:
        print("✗ Failed to install dependencies")
        return False
    return True

def check_model_file():
    """Check if model file exists"""
    model_path = Path("artifacts/runs/hybrid_cqksan_deberta_imdb/best_model.pt")
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✓ Model found: {size_mb:.1f} MB")
        return True
    else:
        print("⚠️  Model file NOT found!")
        print("\nYou need to get the model file from Jagadeesh:")
        print("1. Download best_model.pt from them (via email/WhatsApp/Drive)")
        print(f"2. Place it here: {model_path}")
        print("\nOnce you have the file, run: python app.py --config configs/hybrid.yaml")
        return False

def test_import():
    """Test if all imports work"""
    print("\n🧪 Testing imports...")
    try:
        import torch
        import gradient  # pennylane
        import gradio
        import deberta
        print("✓ All imports working")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def main():
    print("=" * 50)
    print("🚀 Automatic Setup Script")
    print("=" * 50)
    
    # Step 1: Create folders
    create_folders()
    
    # Step 2: Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Step 3: Test imports
    test_import()
    
    # Step 4: Check for model
    has_model = check_model_file()
    
    print("\n" + "=" * 50)
    if has_model:
        print("✅ READY TO RUN!")
        print("=" * 50)
        print("\nRun this to start the app:")
        print("  python app.py --config configs/hybrid.yaml")
    else:
        print("⚠️  WAITING FOR MODEL FILE")
        print("=" * 50)
        print("\nAfter getting best_model.pt, run:")
        print("  python app.py --config configs/hybrid.yaml")

if __name__ == "__main__":
    main()
