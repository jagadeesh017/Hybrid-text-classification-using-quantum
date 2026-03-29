from __future__ import annotations

from typing import Iterable

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def compute_classification_metrics(predictions: Iterable[int], labels: Iterable[int]) -> dict:
    preds = list(predictions)
    gold = list(labels)
    accuracy = accuracy_score(gold, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold, preds, average="weighted", zero_division=0
    )
    matrix = confusion_matrix(gold, preds).tolist()
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": matrix,
    }
