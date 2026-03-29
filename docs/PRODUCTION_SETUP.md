# Production Setup and Deployment Guide

## Pre-Deployment Checklist

This guide ensures your CQKSAN-DeBERTa hybrid quantum NLP system is production-ready.

### 1. Environment Validation

Before running any training or inference, validate your setup:

```bash
python validate_setup.py
```

This script checks:
- Python version (3.8+)
- Required packages installation
- GPU availability (optional but recommended)
- Configuration files integrity
- Source code structure
- Output directories
- Proper file permissions

### 2. Installation and Setup

#### Step 1: Clone/Prepare Repository

```bash
cd hybrid-quantum-nlp
```

#### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n hqnlp python=3.10
conda activate hqnlp
```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 4: Validate Setup

```bash
python validate_setup.py
```

Expected output:
```
✓ PASS: Python Version
✓ PASS: Required Packages
✓ PASS: GPU Availability (or warning if no GPU)
✓ PASS: Configuration Files
✓ PASS: Source Structure
✓ PASS: Main Scripts
✓ PASS: Output Directories
```

### 3. Running Training

#### Basic Training

```bash
python train.py --config configs/hybrid.yaml
```

#### Custom Configuration

```bash
python train.py --config configs/custom.yaml
```

#### Expected Output

```
2026-03-29 10:15:30 - root - INFO - Starting training script
2026-03-29 10:15:31 - hqnlp.config - INFO - Loaded config from: configs/hybrid.yaml
2026-03-29 10:15:32 - hqnlp.data - INFO - Loaded dataset: imdb
...
Training Summary:
  Best f1: 0.9234
  Output directory: artifacts/runs/hybrid_cqksan_deberta_imdb
```

**Training logs:** `training.log`
**Checkpoint location:** `artifacts/runs/hybrid_cqksan_deberta_imdb/best_model.pt`

### 4. Running Inference

#### CLI Inference

```bash
python inference.py \
  --config configs/hybrid.yaml \
  --checkpoint artifacts/runs/hybrid_cqksan_deberta_imdb/best_model.pt \
  --text "This movie was absolutely fantastic!"
```

Expected output:

```
============================================================
INPUT: This movie was absolutely fantastic!
PREDICTION: positive
CONFIDENCE: 94.32%
PROBABILITIES: [0.0568, 0.9432]
============================================================
```

#### Inference Logs

Logs are saved to `inference.log` for debugging.

### 5. Running the Web UI (Gradio)

#### Launch Local Demo

```bash
python app.py \
  --config configs/hybrid.yaml \
  --checkpoint artifacts/runs/hybrid_cqksan_deberta_imdb/best_model.pt \
  --server-port 7860
```

Then open: `http://127.0.0.1:7860`

#### Remote Server Deployment

```bash
python app.py \
  --config configs/hybrid.yaml \
  --checkpoint artifacts/runs/hybrid_cqksan_deberta_imdb/best_model.pt \
  --server-name 0.0.0.0 \
  --server-port 8000
```

Access at: `http://<your-server-ip>:8000`

### 6. Troubleshooting

#### Problem: "Module not found" error

**Solution:**
```bash
python validate_setup.py  # Check if packages are installed
pip install -r requirements.txt  # Reinstall if needed
```

#### Problem: "Config file not found"

**Solution:**
```bash
# Check config path
ls -la configs/
# Update path if needed
python train.py --config ./configs/hybrid.yaml
```

#### Problem: "Checkpoint not found"

**Solution:**
```bash
# Verify checkpoint exists
ls -la artifacts/runs/hybrid_cqksan_deberta_imdb/best_model.pt
# Use absolute path if relative doesn't work
python inference.py \
  --config configs/hybrid.yaml \
  --checkpoint /full/path/to/best_model.pt \
  --text "Your text"
```

#### Problem: "CUDA out of memory"

**Solutions:**
1. Reduce batch size in config:
   ```yaml
   training:
     batch_size: 4  # Decrease from 8
   ```
2. Use CPU if GPU memory insufficient:
   - Script automatically falls back to CPU

#### Problem: "Slow training / inference"

**Solutions:**
1. Verify GPU is being used:
   ```bash
   python validate_setup.py  # Check GPU status
   ```
2. Monitor GPU usage:
   ```bash
   nvidia-smi  # For NVIDIA GPUs
   ```
3. Reduce model size in config:
   ```yaml
   model:
     projector_hidden_dim: 64  # Decrease from 128
   ```

#### Problem: Data loading errors

**Solutions:**
```bash
# Clear cache and re-download
rm -rf artifacts/cache/*
python train.py --config configs/hybrid.yaml
```

#### Problem: Training interrupted / KeyboardInterrupt

**Recovery:**
- Rerun training with same config - best model is saved
- Check training.log for specific errors

### 7. Model Configuration

All configs are in `configs/`:

#### Baseline (DeBERTa only)
```bash
python train.py --config configs/baseline.yaml
```

#### Reduced (DeBERTa + dimensionality reduction)
```bash
python train.py --config configs/reduced.yaml
```

#### Hybrid (DeBERTa + CQKSAN quantum module)
```bash
python train.py --config configs/hybrid.yaml
```

#### Default (Same as hybrid)
```bash
python train.py --config configs/default.yaml
```

### 8. Output and Artifacts

After training, artifacts are saved in `artifacts/runs/<experiment-name>/`:

```
artifacts/runs/hybrid_cqksan_deberta_imdb/
├── best_model.pt          # Best checkpoint
├── config.json            # Training config
├── history.json           # Training history
├── summary.json           # Summary metrics
└── tokenizer/             # Saved tokenizer
    ├── vocab.txt
    ├── config.json
    └── ...
```

### 9. Performance Monitoring

#### Check training progress
```bash
tail -f training.log
```

#### View results
```bash
cat artifacts/runs/hybrid_cqksan_deberta_imdb/summary.json
```

#### Analyze metrics
```bash
python -c "import json; print(json.load(open('artifacts/runs/hybrid_cqksan_deberta_imdb/history.json')))"
```

### 10. Production Deployment Checklist

- [ ] All validation checks pass (`python validate_setup.py`)
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Model trained: `python train.py --config configs/hybrid.yaml`
- [ ] Checkpoint verified: Confirm best_model.pt exists
- [ ] Inference tested: Run sample inference with test text
- [ ] Web UI tested: Verify Gradio app loads and responds
- [ ] Logging enabled: Check training.log and inference.log created
- [ ] GPU/CPU selection: Verify device selection in logs
- [ ] Error handling: Test with invalid inputs, verify graceful errors
- [ ] Documentation: Review DEMO_SCRIPT.md for presentation

### 11. Demo Presentation Commands

```bash
# Check DEMO_SCRIPT.md for full presentation flow
cat docs/DEMO_SCRIPT.md

# Quick demo
python app.py --config configs/hybrid.yaml --checkpoint artifacts/runs/hybrid_cqksan_deberta_imdb/best_model.pt
```

### 12. Emergency Contacts/Debugging

- Check `training.log` and `inference.log` for detailed error traces
- Validate config with: `python validate_setup.py`
- Test individual components:
  ```bash
  python -c "from hqnlp import load_config; print(load_config('configs/hybrid.yaml'))"
  ```

### 13. Notes

- First run may take time to download models and datasets
- GPU strongly recommended for training (10x speedup)
- Inference is fast even on CPU (<100ms per run)
- All commands assume you're in project root directory

---

**For additional details, see:**
- [README.md](README.md) - Project overview
- [FINAL_SUBMISSION.md](docs/FINAL_SUBMISSION.md) - Submission guide
- [DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) - Demo presentation
