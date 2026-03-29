#!/usr/bin/env python
"""
Startup validation script to check environment and dependencies.

Run this before training or inference to ensure all dependencies are installed
and configured correctly.
"""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_python_version():
    """Check if Python version is supported."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error(f"Python 3.8+ required, got {version.major}.{version.minor}")
        return False
    logger.info(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_required_packages():
    """Check if all required packages are installed."""
    required = {
        'torch': '2.1.0',
        'transformers': '4.38.0',
        'datasets': '2.18.0',
        'pennylane': '0.37.0',
        'scikit-learn': '1.4.0',
        'gradio': '5.0.0',
        'tqdm': '4.66.0',
        'pyyaml': '6.0.1',
    }
    
    missing = []
    for package, min_version in required.items():
        try:
            mod = __import__(package.replace('-', '_'))
            version = getattr(mod, '__version__', 'unknown')
            logger.info(f"✓ {package}: {version}")
        except ImportError:
            logger.error(f"✗ {package}: NOT INSTALLED")
            missing.append(package)
    
    if missing:
        logger.error(f"\nMissing packages: {', '.join(missing)}")
        logger.error("Install with: pip install -r requirements.txt")
        return False
    return True


def check_gpu_availability():
    """Check if GPU is available (optional but recommended)."""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "unknown"
            logger.info(f"✓ GPU available: {device_name} (count: {device_count})")
            return True
        else:
            logger.warning("⚠ No GPU detected. Training will use CPU (slower).")
            return True
    except Exception as e:
        logger.warning(f"⚠ Could not check GPU: {e}")
        return True


def check_config_files():
    """Check if required config files exist."""
    configs = [
        "configs/default.yaml",
        "configs/hybrid.yaml",
        "configs/baseline.yaml",
        "configs/reduced.yaml",
    ]
    
    missing = []
    for config_path in configs:
        path = Path(config_path)
        if path.exists():
            logger.info(f"✓ {config_path}")
        else:
            logger.error(f"✗ {config_path}: NOT FOUND")
            missing.append(config_path)
    
    if missing:
        logger.error(f"Missing config files: {missing}")
        return False
    return True


def check_source_structure():
    """Check if source structure is valid."""
    required_dirs = [
        "src/hqnlp",
        "src/hqnlp/data",
        "src/hqnlp/models",
        "src/hqnlp/training",
        "src/hqnlp/inference",
        "src/hqnlp/evaluation",
        "src/hqnlp/ui",
    ]
    
    missing = []
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            logger.info(f"✓ {dir_path}")
        else:
            logger.error(f"✗ {dir_path}: NOT FOUND")
            missing.append(dir_path)
    
    if missing:
        logger.error(f"Missing directories: {missing}")
        return False
    return True


def check_main_scripts():
    """Check if main entry points exist."""
    scripts = [
        "train.py",
        "inference.py",
        "app.py",
    ]
    
    missing = []
    for script in scripts:
        path = Path(script)
        if path.exists():
            logger.info(f"✓ {script}")
        else:
            logger.error(f"✗ {script}: NOT FOUND")
            missing.append(script)
    
    if missing:
        logger.error(f"Missing scripts: {missing}")
        return False
    return True


def check_output_directories():
    """Check and create output directories."""
    output_dirs = [
        "artifacts/runs",
        "artifacts/cache",
    ]
    
    for dir_path in output_dirs:
        path = Path(dir_path)
        try:
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ {dir_path} (ready)")
        except Exception as e:
            logger.error(f"✗ Failed to create {dir_path}: {e}")
            return False
    
    return True


def main():
    """Run all validation checks."""
    logger.info("=" * 70)
    logger.info("CQKSAN-DeBERTa Hybrid Quantum NLP - Startup Validation")
    logger.info("=" * 70)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_required_packages),
        ("GPU Availability", check_gpu_availability),
        ("Configuration Files", check_config_files),
        ("Source Structure", check_source_structure),
        ("Main Scripts", check_main_scripts),
        ("Output Directories", check_output_directories),
    ]
    
    results = []
    for name, check_func in checks:
        logger.info(f"\nChecking {name}...")
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"Error checking {name}: {e}")
            results.append((name, False))
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("Validation Summary:")
    logger.info("=" * 70)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {name}")
        if not result:
            all_passed = False
    
    logger.info("=" * 70)
    
    if all_passed:
        logger.info("✓ All checks passed! Setup is ready.")
        logger.info("\nYou can now run:")
        logger.info("  - python train.py --config configs/hybrid.yaml")
        logger.info("  - python inference.py --config configs/hybrid.yaml --checkpoint <path> --text '<text>'")
        logger.info("  - python app.py --config configs/hybrid.yaml --checkpoint <path>")
        return 0
    else:
        logger.error("✗ Some checks failed. See errors above.")
        logger.error("\nCommon fixes:")
        logger.error("  1. Install dependencies: pip install -r requirements.txt")
        logger.error("  2. Check you're in the project root directory")
        logger.error("  3. Ensure all config files are present")
        return 1


if __name__ == "__main__":
    sys.exit(main())
