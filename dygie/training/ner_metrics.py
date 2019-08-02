from overrides import overrides
from typing import Optional, List, Dict

from allennlp.training.metrics.metric import Metric
import pandas as pd

from dygie.training.f1 import compute_f1


class NERMetrics(Metric):
    """
    Computes precision, recall, and micro-averaged F1 from a list of predicted and gold labels.
    """

    def __init__(self, entity_classes):
        self._classes = entity_classes
        self.reset()

    @overrides
    def __call__(self, predictions: List[str], gold_labels: List[str]):
        for p, g in zip(predictions, gold_labels):
            self._confusion_matrix.at[g, p] += 1

    def get_metric(self, reset: bool = False):
        all_tags = set(self._classes)
        all_metrics = {}
        total_matched = self.filter_null_label({k: self._confusion_matrix.at[k, k] for k in self._classes})
        total_predicted = self.filter_null_label(self._confusion_matrix.sum(0).to_dict())
        total_gold = self.filter_null_label(self._confusion_matrix.sum(1).to_dict())

        for tag in all_tags:
            if tag != "":
                precision, recall, f1_measure = compute_f1(total_predicted[tag], total_gold[tag], total_matched[tag])
                precision_key = "precision" + "-" + tag
                recall_key = "recall" + "-" + tag
                f1_key = "f1-measure" + "-" + tag
                all_metrics[precision_key] = precision
                all_metrics[recall_key] = recall
                all_metrics[f1_key] = f1_measure

        # Compute the precision, recall and f1 for all spans jointly.
        precision, recall, f1_measure = compute_f1(
            sum(total_predicted.values()), sum(total_gold.values()), sum(total_matched.values())
        )
        all_metrics["precision"] = precision
        all_metrics["recall"] = recall
        all_metrics["f1"] = f1_measure
        if reset:
            print(self._confusion_matrix.to_string())
            self.reset()
        return all_metrics

    @staticmethod
    def filter_null_label(metric):
        return {k: v for k, v in metric.items() if k != ""}

    def reset(self):
        self._confusion_matrix = pd.DataFrame.from_dict(
            {gold: {pred: 0 for pred in self._classes} for gold in self._classes}, orient="index"
        )