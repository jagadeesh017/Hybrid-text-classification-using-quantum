# How to Run the Project from GitHub 🚀

Yes! We **did** make a Gradio app. Here's why it wasn't working and how to fix it.

---

## The Problem ❌

The Gradio app needs a **trained model checkpoint** to work. We haven't trained any models yet, so there's no checkpoint file.

**Error you might see:**
```
ValueError: A checkpoint path is required to launch the app.
```

---

## The 5-Step Solution ✅

### Step 1: Clone from GitHub

```bash
git clone https://github.com/jagadeesh017/Hybrid-text-classification-using-quantum.git
cd Hybrid-text-classification-using-quantum
```

### Step 2: Set Up Python Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Verify installation:**
```bash
python validate_setup.py
```

You should see: ✓ All checks pass

### Step 4: Train a Model (Required!)

Choose ONE of these options:

**Option A: Quick Training (5-10 minutes on GPU, 20-30 min on CPU)**
```bash
python train.py --config configs/hybrid.yaml
```

**Option B: Train All Models for Comparison (20-40 minutes total)**
```bash
python run_experiments.py --all
```

**What happens:**
- Model downloads DeBERTa (first time only, ~350MB)
- Downloads IMDB dataset (first time only, ~80MB)
- Trains for ~3-5 epochs
- Saves checkpoint to: `artifacts/runs/hybrid_cqksan_deberta_imdb/best_model.pt`

### Step 5: Launch the Gradio Web App! 🎉

```bash
python app.py --config configs/hybrid.yaml
```

**You should see:**
```
Launching app at 127.0.0.1:7860
```

**Open browser:**
- Go to: http://127.0.0.1:7860
- See the Gradio interface
- Type text and click "Run inference"
- Get predictions!

---

## Quick Test Without Training (Demo Mode)

If you want to test the interface WITHOUT training first:

```bash
# This will show you the UI structure
python -c "
from pathlib import Path
import sys
ROOT = Path('.').resolve()
SRC = ROOT / 'src'
sys.path.insert(0, str(SRC))
from hqnlp.ui.app import build_demo
demo = build_demo('configs/hybrid.yaml', None)
print('Gradio app built successfully!')
print('To launch: python app.py --config configs/hybrid.yaml')
"
```

---

## Why Training is Needed 📚

The Gradio app architecture:

```
User Input Text
        ↓
    [Gradio UI]
        ↓
   app.py loads:
   - Config file (configs/hybrid.yaml)
   - Checkpoint (artifacts/runs/.../best_model.pt) ← THIS MUST EXIST!
        ↓
   Runs inference on text
        ↓
   Returns: Prediction + Confidence
```

**Checkpoint file path:**
```
artifacts/runs/hybrid_cqksan_deberta_imdb/best_model.pt
                    ↑
            Created after training
```

---

## Complete Workflow Example

```bash
# 1. Clone
git clone https://github.com/jagadeesh017/Hybrid-text-classification-using-quantum.git
cd Hybrid-text-classification-using-quantum

# 2. Setup (Windows)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python validate_setup.py

# 3. Train (this takes 10-30 minutes)
python train.py --config configs/hybrid.yaml

# 4. Launch Gradio app
python app.py --config configs/hybrid.yaml

# 5. Open browser
# → http://127.0.0.1:7860
# → Type: "This movie was amazing"
# → Click "Run inference"
# → See prediction!
```

---

## Troubleshooting 🔧

### Problem: "Module not found" error

**Solution:**
```bash
python validate_setup.py
pip install -r requirements.txt
```

### Problem: "Checkpoint not found"

**Solution:**
You need to train first. Run:
```bash
python train.py --config configs/hybrid.yaml
```

### Problem: Out of memory during training

**Solution:**
Edit `configs/hybrid.yaml`:
```yaml
training:
  batch_size: 4    # Change from 8 to 4
```

Then train again:
```bash
python train.py --config configs/hybrid.yaml
```

### Problem: Training is slow

**Normal!** 
- CPU: 20-40 minutes
- GPU: 5-10 minutes
- First time downloads ~430MB of data

### Problem: Gradio app won't start

**Check:**
1. Did training complete successfully?
   ```bash
   ls artifacts/runs/hybrid_cqksan_deberta_imdb/best_model.pt
   ```

2. Are dependencies installed?
   ```bash
   python validate_setup.py
   ```

3. Is port 7860 already in use?
   ```bash
   python app.py --server-port 8000
   ```

---

## Other Ways to Use the Project

### Inference Only (No UI)

```bash
python inference.py \
  --config configs/hybrid.yaml \
  --checkpoint artifacts/runs/hybrid_cqksan_deberta_imdb/best_model.pt \
  --text "This is a great movie!"
```

Output:
```
PREDICTION: positive
CONFIDENCE: 94.32%
```

### Compare All Models

```bash
python run_experiments.py --all
cat results/RESULTS.md
```

### Run Tests

```bash
pytest tests/ -v
```

---

## File Structure After Training

```
hybrid-quantum-nlp/
├── app.py                          # Gradio app launcher ← USE THIS!
├── train.py
├── inference.py
├── validate_setup.py
├── requirements.txt
├── configs/
│   ├── hybrid.yaml
│   ├── baseline.yaml
│   └── reduced.yaml
└── artifacts/
    └── runs/
        └── hybrid_cqksan_deberta_imdb/
            ├── best_model.pt       # ← Created after training
            ├── config.json
            ├── history.json
            └── summary.json
```

---

## Summary

| Task | Command | Time |
|------|---------|------|
| Clone | `git clone ...` | 1 min |
| Setup | `pip install -r requirements.txt` | 2-5 min |
| **Train Model** | `python train.py --config configs/hybrid.yaml` | **10-30 min** ⏳ |
| Launch App | `python app.py --config configs/hybrid.yaml` | 30 sec |
| **Use App** | Open http://127.0.0.1:7860 | ⏱️ Real-time |

**Total time to working app: 15-40 minutes**

---

## Next Steps

1. ✅ Follow the 5-Step Solution above
2. ✅ Wait for training to complete
3. ✅ Open Gradio app in browser
4. ✅ Test with sample text
5. ✅ See predictions!

You've got this! 🚀
