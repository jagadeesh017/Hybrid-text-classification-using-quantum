from __future__ import annotations

import argparse
import logging
import sys

from hqnlp import load_config
from hqnlp.data import build_dataloaders
from hqnlp.models import build_model
from hqnlp.training.trainer import Trainer

logger = logging.getLogger(__name__)


def main():
    """Train a model with error handling."""
    parser = argparse.ArgumentParser(description="Train a CQKSAN-DeBERTa experiment.")
    parser.add_argument("--config", default="configs/default.yaml", help="Config YAML file path")
    args = parser.parse_args()

    try:
        logger.info(f"Loading config from: {args.config}")
        config = load_config(args.config)
        logger.info(f"Experiment: {config.experiment.name}")
        logger.info(f"Model: {config.model.model_type}, Dataset: {config.data.dataset_name}")
        
        logger.info("Building data loaders...")
        train_loader, eval_loader, tokenizer, num_labels, label_names = build_dataloaders(
            config.data, config.model, config.training
        )
        config.model.num_labels = num_labels
        if not config.inference.class_names:
            config.inference.class_names = label_names
        logger.info(f"Label names: {label_names}")

        logger.info(f"Building {config.model.model_type} model...")
        model = build_model(config.model)
        
        logger.info("Initializing trainer...")
        trainer = Trainer(config, model, train_loader, eval_loader, tokenizer, label_names)
        
        logger.info("Starting training...")
        summary = trainer.fit()
        
        logger.info("Training finished.")
        logger.info(f"Best {config.training.metric_for_best_model}: {summary['best_metric']:.4f}")
        logger.info(f"Artifacts saved to: {summary['output_dir']}")
        print("\n" + "="*60)
        print(f"Training Summary:")
        print(f"  Best {config.training.metric_for_best_model}: {summary['best_metric']:.4f}")
        print(f"  Output directory: {summary['output_dir']}")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
