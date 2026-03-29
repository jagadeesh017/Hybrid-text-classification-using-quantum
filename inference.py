import argparse
import logging
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hqnlp import load_config
from hqnlp.data import resolve_label_names
from hqnlp.inference.predict import predict_text, InferenceError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('inference.log')
    ]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run CQKSAN-DeBERTa inference.")
    parser.add_argument("--config", default="configs/default.yaml", help="Config YAML file path")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--text", required=True, help="Text to classify")
    args = parser.parse_args()

    try:
        logger.info(f"Loading config from: {args.config}")
        config = load_config(args.config)
        logger.info(f"Dataset: {config.data.dataset_name}, Model: {config.model.model_type}")
        
        label_names = resolve_label_names(config.data.dataset_name)
        result = predict_text(args.text, config, args.checkpoint, label_names)
        
        print(f"\n{'='*60}")
        print(f"INPUT: {args.text[:100]}...")
        print(f"PREDICTION: {result['label']}")
        print(f"CONFIDENCE: {result['confidence'] * 100:.2f}%")
        print(f"PROBABILITIES: {result['probabilities']}")
        print(f"{'='*60}\n")
        logger.info(f"Inference successful: {result['label']}")
    except InferenceError as e:
        logger.error(f"Inference error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
