import logging
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hqnlp import load_config, build_model
from hqnlp.data import resolve_label_names, build_dataloaders
from hqnlp.training import Trainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train CQKSAN-DeBERTa model")
    parser.add_argument("--config", default="configs/hybrid.yaml", help="Path to config file")
    args = parser.parse_args()
    
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    logger.info("Loading dataset and creating data loaders...")
    train_loader, eval_loader, tokenizer, num_labels, label_names = build_dataloaders(config.data, config.model, config.training)
    
    logger.info("Resolving label names...")
    logger.info(f"Label names: {label_names}")
    
    logger.info(f"Building {config.model.model_type} model...")
    model = build_model(config.model)
    logger.info(f"Model created: {config.model.model_type}")
    
    logger.info("Initializing trainer...")
    trainer = Trainer(config, model, train_loader, eval_loader, tokenizer, label_names)
    
    logger.info("Starting training...")
    summary = trainer.fit()
    
    logger.info(f"Training completed!")
    logger.info(f"Best metric: {summary['best_metric']:.4f}")
    logger.info(f"Checkpoint saved to: {summary['output_dir']}")
    
    return summary


if __name__ == "__main__":
    try:
        logger.info("Starting training script")
        main()
        logger.info("Training completed successfully")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
