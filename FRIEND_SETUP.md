# Setup Guide for Your Friend

## 🚀 SIMPLEST WAY (30 seconds)

**Step 1:** Clone repo
```bash
git clone https://github.com/jagadeesh017/Hybrid-text-classification-using-quantum.git
cd hybrid-quantum-nlp
```

**Step 2:** Run auto setup (creates folders + installs everything)
```bash
python setup.py
```

**Step 3:** Get `best_model.pt` from you
- You send the file via email/WhatsApp/Google Drive
- They place it in: `artifacts/runs/hybrid_cqksan_deberta_imdb/best_model.pt`

**Step 4:** Run app
```bash
python app.py --config configs/hybrid.yaml
```

Done! 👍 Open http://127.0.0.1:7860
