"""Inference module for text classification."""

from .predict import InferenceError, load_model_for_inference, predict_text

__all__ = ["predict_text", "load_model_for_inference", "InferenceError"]

