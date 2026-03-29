# Hybrid Quantum-Classical NLP with CQKSAN-DeBERTa

## 📚 What is This Project?

This project demonstrates **text classification** (determining if text is positive/negative sentiment) using a unique **hybrid approach** that combines:
- **Classical AI:** DeBERTa (a powerful language model)
- **Quantum Computing:** CQKSAN (a quantum-powered attention layer)

Think of it like: *"Using both traditional AI AND quantum computing together to understand text better."*

**Status:** ✅ Production Ready  
**Use Case:** Classify text sentiment (movie reviews, product feedback, etc.)  
**GitHub:** https://github.com/jagadeesh017/Hybrid-text-classification-using-quantum

---

## 🎯 What Can You Do?

| Task | Time | Details |
|------|------|---------|
| **Test with Pre-trained Model** | 2 min | Use existing trained model to classify text |
| **Try the Web Interface** | 5 min | Upload text via browser and get instant results |
| **Train Your Own Model** | 24 hours | Build a custom model with your own settings |
| **Compare 3 Approaches** | 24+ hours | Run baseline, reduced, and hybrid models side-by-side |

---

## 🚀 Quick Start (Choose Your Path)

### Path A: Test Pre-trained Model (Fastest - 2 minutes)

**Step 1:** Get the code
```bash
git clone https://github.com/jagadeesh017/Hybrid-text-classification-using-quantum.git
cd Hybrid-text-classification-using-quantum
```

**Step 2:** Install requirements
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate

# Install packages
pip install --upgrade pip
pip install -r requirements.txt
```

**Step 3:** Set up directories (creates folders + checks setup)
```bash
python setup.py
```

**Step 4a:** Launch web interface (browser-based)
```bash
python app.py --config configs/hybrid.yaml
```
→ Open: **http://127.0.0.1:7860** in your browser

**Step 4b:** Or test via command line
```bash
python inference.py \
  --config configs/hybrid.yaml \
  --checkpoint artifacts/runs/hybrid_cqksan_deberta_imdb/best_model.pt \
  --text "This movie was amazing!"
```

✅ **Done!** You're classifying text with a quantum model.

---

### Path B: Train Your Own Model (Takes 24 hours, GPU recommended)

```bash
# 1. Clone and setup (same as above)
git clone https://github.com/jagadeesh017/Hybrid-text-classification-using-quantum.git
cd Hybrid-text-classification-using-quantum
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Train the model
python train.py --config configs/hybrid.yaml

# 3. Launch the app
python app.py --config configs/hybrid.yaml
```

**What happens during training:**
- Downloads 400 movie reviews (IMDB dataset)
- Trains the quantum+classical model
- Saves the trained model to `artifacts/runs/hybrid_cqksan_deberta_imdb/best_model.pt`
- Saves performance metrics and training history

---

### Path C: Compare All 3 Models (Takes 24+ hours)

```bash
python run_experiments.py --all
cat results/RESULTS.md  # View comparison
```

This trains and compares:
1. **Baseline:** Classical AI only (no quantum)
2. **Reduced:** Classical AI + dimensionality reduction
3. **Hybrid:** Classical AI + Quantum computing ⭐

---

## 📖 Understanding the Project

### What are the files?

| File | Purpose | You Need To |
|------|---------|-------------|
| `app.py` | Web interface (Gradio) | Run to launch browser interface |
| `train.py` | Training script | Run to train a new model |
| `inference.py` | Make predictions | Run to test on one text |
| `setup.py` | Environment setup | Run once to prepare folders |
| `validate_setup.py` | Check everything works | Run if you get errors |
| `requirements.txt` | Package dependencies | Let pip install from this |

### What are the folders?

```
├── configs/          → Settings for training (batch size, learning rate, etc.)
├── src/hqnlp/        → The actual AI code (models, training, etc.)
├── tests/            → Automated tests to verify everything works
├── artifacts/        → Where trained models get saved
└── results/          → Experiment comparisons
```

### What is a "config"?

A configuration file (YAML) that tells the model:
- How many examples to train on
- What learning rate to use
- How many training rounds (epochs)
- Other settings

**Common configs:**
- `hybrid.yaml` - Full quantum model (best accuracy)
- `baseline.yaml` - Classical only (fastest, baseline)
- `reduced.yaml` - With dimensionality reduction

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
