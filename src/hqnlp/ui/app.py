from __future__ import annotations

import argparse
import logging
import sys

import gradio as gr

from hqnlp import load_config
from hqnlp.data import resolve_label_names
from hqnlp.inference.predict import predict_text, InferenceError

logger = logging.getLogger(__name__)


def build_demo(config_path: str, checkpoint_path: str | None = None):
    """Build Gradio demo with error handling.
    
    Args:
        config_path: Path to config YAML file
        checkpoint_path: Optional checkpoint path override
        
    Returns:
        gr.Blocks: Gradio demo interface
        
    Raises:
        ValueError: If configuration or checkpoint is invalid
    """
    try:
        config = load_config(config_path)
        checkpoint = checkpoint_path or config.inference.checkpoint_path
        if not checkpoint:
            raise ValueError("A checkpoint path is required to launch the app.")
        class_names = resolve_label_names(config.data.dataset_name)
    except Exception as e:
        logger.error(f"Failed to load config or checkpoint: {e}")
        raise

    def classify(text: str):
        """Classify input text with error handling.
        
        Args:
            text: Input text to classify
            
        Returns:
            dict: Prediction result or error message
        """
        if not text or not text.strip():
            return {"error": "Enter some text to classify."}
        try:
            result = predict_text(text, config, checkpoint, class_names)
            return {
                "prediction": result["label"],
                "confidence": round(result["confidence"] * 100, 2),
                "probabilities": {
                    class_names[idx] if idx < len(class_names) else str(idx): round(score, 4)
                    for idx, score in enumerate(result["probabilities"])
                },
            }
        except InferenceError as e:
            logger.error(f"Inference error: {e}")
            return {"error": f"Prediction failed: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error during classification: {e}")
            return {"error": "An unexpected error occurred during classification."}

    with gr.Blocks(title="CQKSAN DeBERTa Text Classifier") as demo:
        gr.Markdown(
            """
            # CQKSAN-DeBERTa Text Classification
            Hybrid quantum-classical text classification with DeBERTa embeddings and CQKSAN attention.
            """
        )
        with gr.Row():
            text = gr.Textbox(label="Input text", lines=6, placeholder="Paste a review, news snippet, or SMS here.")
            output = gr.JSON(label="Prediction")
        button = gr.Button("Run inference")
        button.click(classify, inputs=text, outputs=output)
        text.submit(classify, inputs=text, outputs=output)
    return demo


def main():
    """Launch the Gradio app with validation."""
    parser = argparse.ArgumentParser(description="Launch the CQKSAN Gradio app.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML file")
    parser.add_argument("--checkpoint", default="", help="Path to model checkpoint")
    parser.add_argument("--server-name", default="127.0.0.1", help="Server address")
    parser.add_argument("--server-port", type=int, default=7860, help="Server port")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    try:
        demo = build_demo(args.config, args.checkpoint or None)
        logger.info(f"Launching app at {args.server_name}:{args.server_port}")
        demo.launch(server_name=args.server_name, server_port=args.server_port, share=False)
    except Exception as e:
        logger.error(f"Failed to launch app: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
