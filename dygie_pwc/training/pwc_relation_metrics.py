from overrides import overrides
from typing import List, Any

from allennlp.training.metrics.metric import Metric

from dygie_pwc.training.f1 import compute_f1
import numpy as np

class RelationMetrics(Metric):
    """
    Computes precision, recall, and micro-averaged F1 from a list of predicted and gold spans.
    """

    def __init__(self, labels: List[Any]):
        self._labels = labels
        self.reset()

    @overrides
    def __call__(self, predicted_relation_list, gold_relation_list):
        for predicted_relations, gold_relations in zip(predicted_relation_list, gold_relation_list):
            for k, v in gold_relations.items():
                self._total_gold[v] += 1

            for k, v in predicted_relations.items():
                self._total_predicted[v] += 1
                if k in gold_relations:
                    assert gold_relations[k] == v
                    self._total_matched[v] += 1

    @overrides
    def get_metric(self, reset=False):
        precision, recall, f1 = {}, {}, {}
        for k in self._labels:
            precision[k], recall[k], f1[k] = compute_f1(
                self._total_predicted[k], self._total_gold[k], self._total_matched[k]
            )

        precision["micro"], recall["micro"], f1["micro"] = compute_f1(
            sum(list(self._total_predicted.values())),
            sum(list(self._total_gold.values())),
            sum(list(self._total_matched.values())),
        )

        precision["macro"], recall["macro"], f1["macro"] = (
            np.mean(list(precision.values())),
            np.mean(list(recall.values())),
            np.mean(list(f1.values())),
        )

        # Reset counts if at end of epoch.
        if reset:
            self.reset()

        return precision, recall, f1

    @overrides
    def reset(self):
        self._total_gold = {k: 0 for k in self._labels}
        self._total_predicted = {k: 0 for k in self._labels}
        self._total_matched = {k: 0 for k in self._labels}
