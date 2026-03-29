# Setup Guide for Collaborators

## Quick Setup (5 minutes)

### Step 1: Clone the Repository
```bash
git clone https://github.com/jagadeesh017/Hybrid-text-classification-using-quantum.git
cd hybrid-quantum-nlp
```

### Step 2: Get the Model File
**Ask Jagadeesh to send you `best_model.pt` file** (via email/Google Drive/WhatsApp)

### Step 3: Place Model in Correct Location
Create the folder structure and place the file:
```
artifacts/
└── runs/
    └── hybrid_cqksan_deberta_imdb/
        └── best_model.pt  ← Place the file here
```

Windows PowerShell:
```powershell
New-Item -ItemType Directory -Path "artifacts/runs/hybrid_cqksan_deberta_imdb" -Force
# Then paste best_model.pt into this folder
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

**Required:** Python 3.8 or higher

### Step 5: Run the App
```bash
python app.py --config configs/hybrid.yaml
```

Open your browser to: **http://127.0.0.1:7860**

✅ **Done! You can now classify text with the quantum model**

---

## Troubleshooting

**"No such file: best_model.pt"**
→ Make sure the .pt file is in `artifacts/runs/hybrid_cqksan_deberta_imdb/` folder

**"Module not found" errors**
→ Run: `pip install -r requirements.txt`

**Port 7860 already in use**
→ The app will automatically use another port - check the terminal output

---

## Optional: Train Your Own Model (24 hours)

If you want to train instead of using the pre-trained model:
```bash
python train.py --config configs/hybrid.yaml
```

**Note:** This takes ~24 hours on GPU. Skip this if using the provided model.
