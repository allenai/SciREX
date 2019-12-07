"""
Function to compute F1 scores.
"""
from sklearn.metrics import classification_report
import numpy as np


def safe_div(num, denom, m=100):
    if denom > 0 and num >= 0:
        return num * m / denom
    else:
        return -1


def compute_f1(predicted, gold, matched, m=100):
    precision = safe_div(matched, predicted, m)
    recall = safe_div(matched, gold, m)
    f1 = safe_div(2 * precision * recall, precision + recall, m=1)
    return precision, recall, f1


def compute_threshold(predicted_scores: np.ndarray, gold: np.ndarray, bins: int = 100) -> float:
    best_threshold = 0.5
    best_value = 0.0
    for threshold in np.linspace(0.001, 0.999, bins):
        predicted = (predicted_scores > threshold).astype(int)
        metrics = classification_report(gold, predicted, output_dict=True)
        if metrics["macro avg"]["f1-score"] > best_value:
            best_value = metrics["macro avg"]["f1-score"]
            best_threshold = threshold

    return best_threshold
