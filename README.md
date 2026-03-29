# Hybrid Quantum-Classical NLP with CQKSAN-DeBERTa

**Text classification** using quantum + classical AI combined.

**Status:** ✅ Model Trained (271 MB, 88.5% accuracy on IMDB)  
**Repo:** https://github.com/jagadeesh017/Hybrid-text-classification-using-quantum

---

## 🚀 Quick Start (5 minutes)

```bash
# 1. Clone
git clone https://github.com/jagadeesh017/Hybrid-text-classification-using-quantum.git
cd Hybrid-text-classification-using-quantum

# 2. Setup
python -m venv venv
venv\Scripts\activate          # Windows: venv\Scripts\activate | macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python setup.py

# 3. Get model (download best_model.pt from GitHub Releases)
# Place in: artifacts/runs/hybrid_cqksan_deberta_imdb/best_model.pt

# 4. Run
python app.py --config configs/hybrid.yaml
```

Open: **http://127.0.0.1:7860**

---

## 📋 Team Workflow

### Clone + Setup
```bash
git clone ...
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
python setup.py
```

### Create Branch
```bash
git checkout -b feature/your-name
```

### Commit & Push
```bash
git add .
git commit -m "type: description"  # feat, fix, refactor, docs, test
git push origin feature/your-name
```

### Create PR
1. Go to GitHub
2. Click "Compare & pull request"
3. Fill title & description
4. Submit

### Before PR
```bash
pytest tests/ -v                # Run tests
python validate_setup.py        # Validate
```

---

## 📁 File Structure

```
src/hqnlp/
  ├── models/          → Quantum & classical models
  ├── training/        → Training loop
  ├── inference/       → Predictions
  ├── data/            → Dataset handling
  └── evaluation/      → Metrics

configs/               → Training settings (YAML)
tests/                 → Unit tests
artifacts/             → Models (not committed)
```

---

## 🔧 Key Commands

| Command | Purpose |
|---------|---------|
| `python train.py --config configs/hybrid.yaml` | Train model (24h) |
| `python app.py --config configs/hybrid.yaml` | Launch web interface |
| `python inference.py ...` | Test predictions |
| `pytest tests/ -v` | Run tests |
| `python validate_setup.py` | Check setup |

---

## 📥 Model Download

**Pre-trained model available:**  
→ https://github.com/jagadeesh017/Hybrid-text-classification-using-quantum/releases/tag/v1.0-model

Download `best_model.pt` and place in: `artifacts/runs/hybrid_cqksan_deberta_imdb/best_model.pt`

---

## ❌ Don't Commit These

```
*.pyc
__pycache__/
.pytest_cache/
artifacts/              (too large)
.venv/ or venv/
.env
```

Already in `.gitignore` ✓

---

## 🧪 Test Before Submitting

```bash
pytest tests/ -v
python validate_setup.py
```

---

## Project Info

- **Framework:** PyTorch, PennyLane (quantum), Transformers
- **Model:** DeBERTa + 4-qubit CQKSAN
- **Dataset:** IMDB (400 train, 100 eval)
- **Accuracy:** 88.5% (hybrid) vs 86.5% (baseline)

---

## 📚 Documentation

- **What it is?** → See above
- **How to use?** → Quick Start section
- **How to contribute?** → Team Workflow section
- **Architecture?** → Check `src/hqnlp/models/`

---

**Ready to work?** Clone, setup, create branch, and start coding! 🚀


---

## 🧠 How Does It Work? (Simple Explanation)

### Traditional AI Approach:
```
Text → Language Model (DeBERTa) → Classifier → Result
```

### This Project (Quantum Hybrid):
```
Text 
  ↓
Language Model (DeBERTa extracts meaning)
  ↓
Split into 2 paths:
  ├─ Classical Path → Regular AI layer
  └─ Quantum Path → Quantum circuit processes it
  ↓
Fusion Layer (combines both paths)
  ↓
Classifier (decides positive or negative)
  ↓
Result with confidence score
```

**Why do this?** Quantum computing might find patterns classical AI misses (though on simulators, the advantage is experimental).

---

## 📊 What Results Should You Expect?

When you train the model, you'll get:
- **Accuracy:** ~88.5% (correctly classifies ~9 out of 10 reviews)
- **Training Time:** ~6 minutes per epoch (GPU) or ~30 min (CPU)
- **Total Training:** ~5 epochs = 30 min (GPU) or 2.5 hours (CPU)

Pre-trained models are already fully trained and ready to use (no waiting).

---

## ❓ Common Questions

**Q: Do I need a quantum computer?**  
A: No! The code uses a quantum simulator (software). Real quantum hardware optional.

**Q: How much disk space?**  
A: ~2 GB (includes DeBERTa model download)

**Q: Can I use my own dataset?**  
A: Yes! Edit `configs/hybrid.yaml` and change `dataset_name` or point to your data.

**Q: Why does it take 24 hours to train?**  
A: Quantum circuits are computationally expensive to simulate. Classical is faster.

**Q: Can I train on CPU?**  
A: Yes, but it's slow. GPU (NVIDIA/AMD) recommended.

**Q: Can I use this for production?**  
A: Yes! Code is production-ready with error handling and validation.

---

## 📥 Installation Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'torch'`
```bash
# Solution: Install requirements
pip install -r requirements.txt
```

### Issue: `CUDA out of memory`
```bash
# Solution: Reduce batch size in config
# Edit configs/hybrid.yaml
training:
  batch_size: 4  # Change from 8 to 4
```

### Issue: Slow training / progress bar stuck
```bash
# This is normal! Quantum circuits are slow
# GPU training: Wait ~6 min per epoch
# CPU training: Wait ~30 min per epoch
```

### Issue: Port 7860 already in use
```bash
# Solution: Gradio auto-finds alternate port
# Check terminal output for actual port (e.g., 7861)
```

---

## 🔧 Advanced Usage

### Custom Training

**Edit the training config:**
```yaml
# configs/my_config.yaml
data:
  dataset_name: imdb        # Change dataset
  num_train: 400            # Change number of examples
  num_eval: 100

model:
  hidden_dim: 128           # Change model size
  dropout: 0.2

training:
  epoch: 5                  # Number of training rounds
  batch_size: 8             # Examples per batch
  learning_rate: 5e-4       # How much to update weights
```

**Run with custom config:**
```bash
python train.py --config configs/my_config.yaml
```

### Custom Inference

```bash
python inference.py \
  --config configs/hybrid.yaml \
  --checkpoint artifacts/runs/hybrid_cqksan_deberta_imdb/best_model.pt \
  --text "This is a great product!"
```

### Batch Inference

```python
from src.hqnlp.inference.predict import Predictor

predictor = Predictor(config_path="configs/hybrid.yaml", 
                     checkpoint_path="path/to/model.pt")

texts = ["Text 1", "Text 2", "Text 3"]
predictions = predictor.predict_batch(texts)
```

---

## 🧪 Testing

**Run all tests:**
```bash
pytest tests/ -v
```

**Run specific test:**
```bash
pytest tests/test_config.py -v
```

---

## 📚 Key Concepts Explained

| Term | Meaning | Example |
|------|---------|---------|
| **DeBERTa** | Pre-trained language model | Understands English like a human |
| **CQKSAN** | Quantum attention layer | Uses quantum circuits to focus on important parts |
| **Epoch** | One complete pass through training data | "The model saw all examples once" |
| **Batch Size** | Examples processed at once | 8 = Process 8 reviews simultaneously |
| **Checkpoint** | Saved model weights | The trained model file |
| **Inference** | Making predictions | "Tell me if this review is positive" |
| **Configuration** | Settings file | "Train with these parameters" |

---

## 📊 Project Statistics

- **Lines of Code:** ~2000
- **Unit Tests:** 24 test cases
- **Models:** 3 variants (baseline, reduced, hybrid)
- **Training Data:** 400 examples (IMDB reviews)
- **Evaluation Data:** 100 examples
- **Quantum Qubits:** 4-qubit VQC
- **Accuracy:** ~88.5% (hybrid) vs 86.5% (baseline)

---

## 🎓 Learning Resources

If concepts are unfamiliar:
- **Deep Learning Basics:** See `docs/DEMO_SCRIPT.md`
- **Quantum Computing:** PennyLane docs (https://pennylane.ai)
- **Transformers:** Hugging Face docs (https://huggingface.co)
- **Gradio UI:** Gradio docs (https://gradio.app)

---

## 🤝 Working with Others

### For Collaborators:

1. **Clone the repo**
   ```bash
   git clone https://github.com/jagadeesh017/Hybrid-text-classification-using-quantum.git
   cd Hybrid-text-classification-using-quantum
   ```

2. **Install and setup**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python setup.py
   ```

3. **Get pre-trained model** (ask maintainer for `best_model.pt`)

4. **Start working**
   ```bash
   python app.py --config configs/hybrid.yaml
   ```

### Before Submitting Changes:

```bash
# Test your code
pytest tests/ -v

# Validate setup
python validate_setup.py

# Commit changes
git add .
git commit -m "Description of changes"
git push origin main
```

---

## 📝 Citation

If you use this in research:

```bibtex
@software{cqksan_hybrid_nlp_2026,
  title={Hybrid Quantum-Classical NLP with CQKSAN-DeBERTa},
  author={Jagadeesh},
  year={2026},
  url={https://github.com/jagadeesh017/Hybrid-text-classification-using-quantum}
}
```

---

## 📧 Support

**Something not working?**

1. Check "Troubleshooting" section above
2. Run `python validate_setup.py` to diagnose
3. Check `requirements.txt` versions match
4. Try reinstalling: `pip install -r requirements.txt --force-reinstall`

---

## 🎯 Next Steps

- ✅ Run pre-trained model (Path A)
- ✅ Train your own model (Path B)
- ✅ Understand the architecture (How Does It Work? section)
- ✅ Customize configuration (Advanced Usage)
- ✅ Integrate into your project (Custom Inference)

---

**Happy Classifying! 🚀**
