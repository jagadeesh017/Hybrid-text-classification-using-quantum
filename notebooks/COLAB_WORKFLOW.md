# Google Colab GPU Workflow

Use this project in Google Colab with a T4 or better GPU.

## 1. Start a GPU runtime

- Open Colab
- `Runtime` -> `Change runtime type`
- Select `T4 GPU`

## 2. Clone the project

```python
!git clone <your-repo-url>
%cd hybrid-quantum-nlp
```

## 3. Install dependencies

```python
!pip install -r requirements.txt
```

## 4. Train the hybrid model

```python
!python train.py --config configs/default.yaml
```

## 5. Run inference

Replace `<run_name>` with the folder created under `artifacts/runs/`.

```python
!python inference.py \
  --config configs/default.yaml \
  --checkpoint artifacts/runs/<run_name>/best_model.pt \
  --text "This film was emotionally rich and visually stunning."
```

## 6. Launch the app

```python
!python app.py \
  --config configs/default.yaml \
  --checkpoint artifacts/runs/<run_name>/best_model.pt
```

## Notes

- Colab GPU accelerates the DeBERTa and PyTorch parts immediately.
- The PennyLane `default.qubit` backend remains simulation-based.
- For larger experiments, store `artifacts/` on Google Drive.
