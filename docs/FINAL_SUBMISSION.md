# Final Submission Guide

This repository is organized for final academic submission and demo delivery.

## What To Submit

Include these folders and files:

- `src/`
- `configs/`
- `scripts/`
- `notebooks/`
- `tests/`
- `docs/`
- `train.py`
- `inference.py`
- `app.py`
- `README.md`
- `requirements.txt`
- `.gitignore`

## Recommended Final Demo Config

Use:

```bash
python train.py --config configs/hybrid.yaml
```

This produces the main hybrid checkpoint at:

```text
artifacts/runs/hybrid_cqksan_deberta_imdb/best_model.pt
```

## Baseline Ablation Runs

```bash
python train.py --config configs/baseline.yaml
python train.py --config configs/reduced.yaml
python train.py --config configs/hybrid.yaml
```

## Demo Commands

### CLI

```bash
python inference.py --config configs/hybrid.yaml --checkpoint artifacts/runs/hybrid_cqksan_deberta_imdb/best_model.pt --text "This movie was excellent."
```

### Gradio

```bash
python app.py --config configs/hybrid.yaml --checkpoint artifacts/runs/hybrid_cqksan_deberta_imdb/best_model.pt
```

## Deliverables To Capture Before Submission

- Best checkpoint
- Training summary JSON
- Confusion matrix and metrics
- 3-5 demo screenshots
- Final results table in `results/`

## Notes

If you are running on Google Colab, follow `notebooks/COLAB_WORKFLOW.md`.
