from __future__ import annotations

import copy
import logging
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from hqnlp.config import AppConfig
from hqnlp.evaluation.metrics import compute_classification_metrics
from hqnlp.utils import ensure_dir, resolve_device, save_json, set_seed

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config: AppConfig, model: nn.Module, train_loader, eval_loader, tokenizer, label_names: list[str]):
        """Initialize trainer with error checking.
        
        Args:
            config: AppConfig instance
            model: Neural network model
            train_loader: Training data loader
            eval_loader: Evaluation data loader
            tokenizer: Tokenizer for preprocessing
            label_names: List of label names
        """
        if not config or not model or not train_loader or not eval_loader:
            raise ValueError("Config, model, and data loaders are required")
        if not label_names or len(label_names) != config.model.num_labels:
            logger.warning(f"Label names mismatch: {len(label_names)} names vs {config.model.num_labels} labels")
            
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.tokenizer = tokenizer
        self.label_names = label_names
        self.device = resolve_device()
        logger.info(f"Using device: {self.device}")
        
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable_params:
            logger.warning("No trainable parameters found in model!")
            
        self.optimizer = AdamW(
            trainable_params,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        logger.info(f"Optimizer: AdamW (lr={config.training.learning_rate})")

        total_steps = max(1, len(train_loader) * config.training.epochs // config.training.gradient_accumulation_steps)
        warmup_steps = int(total_steps * config.training.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        amp_enabled = config.training.mixed_precision and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        logger.info(f"Mixed precision: {amp_enabled}")
        
        self.output_dir = ensure_dir(Path(config.experiment.output_dir) / config.experiment.name)
        save_json(asdict(config), self.output_dir / "config.json")
        logger.info(f"Output directory: {self.output_dir}")

    def fit(self) -> dict:
        """Train the model with early stopping and checkpoint management.
        
        Returns:
            dict: Training summary with metrics and checkpoint path
        """
        try:
            set_seed(self.config.experiment.seed)
            logger.info(f"Starting training with seed {self.config.experiment.seed}")
            
            best_metric = float("-inf")
            best_state = None
            patience = 0
            history: list[dict] = []

            for epoch in range(1, self.config.training.epochs + 1):
                try:
                    logger.info(f"Epoch {epoch}/{self.config.training.epochs}")
                    train_stats = self._train_epoch(epoch)
                    eval_stats = self.evaluate()
                    epoch_stats = {"epoch": epoch, **train_stats, **eval_stats}
                    history.append(epoch_stats)
                    save_json(history, self.output_dir / "history.json")

                    metric_name = self.config.training.metric_for_best_model
                    current_metric = eval_stats.get(metric_name, float("-inf"))
                    
                    logger.info(f"Epoch {epoch} - Loss: {train_stats['train_loss']:.4f}, "
                              f"{metric_name}: {current_metric:.4f}")
                    
                    if current_metric > best_metric:
                        best_metric = current_metric
                        best_state = copy.deepcopy(self.model.state_dict())
                        torch.save(best_state, self.output_dir / "best_model.pt")
                        logger.info(f"New best {metric_name}: {best_metric:.4f}")
                        patience = 0
                    else:
                        patience += 1
                        logger.info(f"No improvement. Patience: {patience}/{self.config.training.early_stopping_patience}")

                    if patience >= self.config.training.early_stopping_patience:
                        logger.info("Early stopping triggered!")
                        break
                        
                except Exception as e:
                    logger.error(f"Error in epoch {epoch}: {e}", exc_info=True)
                    raise

            if best_state is not None:
                self.model.load_state_dict(best_state)
                logger.info("Loaded best model state")
            else:
                logger.warning("No best state found, using current model")

            final_eval = self.evaluate()
            summary = {
                "best_metric": best_metric,
                "final_eval": final_eval,
                "history": history,
                "output_dir": str(self.output_dir),
            }
            save_json(summary, self.output_dir / "summary.json")
            self.tokenizer.save_pretrained(self.output_dir / "tokenizer")
            logger.info(f"Training complete. Best {metric_name}: {best_metric:.4f}")
            logger.info(f"Artifacts saved to {self.output_dir}")
            return summary
            
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        running_loss = 0.0
        predictions: list[int] = []
        labels: list[int] = []
        start = time.perf_counter()
        accumulation_steps = self.config.training.gradient_accumulation_steps

        progress = tqdm(self.train_loader, desc=f"Train {epoch}/{self.config.training.epochs}")
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(progress, start=1):
            batch = {key: value.to(self.device) for key, value in batch.items()}
            target = batch["labels"]
            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                loss = self.criterion(outputs["logits"], target) / accumulation_steps

            self.scaler.scale(loss).backward()

            if step % accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()

            running_loss += loss.item() * accumulation_steps
            batch_preds = torch.argmax(outputs["logits"], dim=-1)
            predictions.extend(batch_preds.detach().cpu().tolist())
            labels.extend(target.detach().cpu().tolist())
            progress.set_postfix(loss=f"{running_loss / step:.4f}")

        metrics = compute_classification_metrics(predictions, labels)
        metrics["train_loss"] = running_loss / max(1, len(self.train_loader))
        metrics["train_runtime_seconds"] = round(time.perf_counter() - start, 2)
        return metrics

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        predictions: list[int] = []
        labels: list[int] = []
        running_loss = 0.0
        start = time.perf_counter()

        for batch in tqdm(self.eval_loader, desc="Eval", leave=False):
            batch = {key: value.to(self.device) for key, value in batch.items()}
            outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = self.criterion(outputs["logits"], batch["labels"])
            running_loss += loss.item()
            batch_preds = torch.argmax(outputs["logits"], dim=-1)
            predictions.extend(batch_preds.cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())

        metrics = compute_classification_metrics(predictions, labels)
        metrics["eval_loss"] = running_loss / max(1, len(self.eval_loader))
        metrics["eval_runtime_seconds"] = round(time.perf_counter() - start, 2)
        metrics["label_names"] = self.label_names
        return metrics
