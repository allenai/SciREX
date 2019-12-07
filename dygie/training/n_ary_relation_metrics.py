from overrides import overrides
from allennlp.training.metrics.metric import Metric

import pandas as pd
from sklearn.metrics import classification_report

from dygie.training.f1 import compute_threshold


class NAryRelationMetrics(Metric):
    def __init__(self):
        self.reset()

    @overrides
    def __call__(
        self,
        candidate_relation_list,
        candidate_relation_labels,
        candidate_relation_scores,
        doc_id,
    ):
        try:
            candidate_relation_scores, = self.unwrap_to_tensors(
                candidate_relation_scores
            )
            candidate_relation_scores = list(candidate_relation_scores.numpy())
        except:
            breakpoint()

        assert len(candidate_relation_list) == len(
            candidate_relation_scores
        ), breakpoint()
        assert len(candidate_relation_labels) == len(
            candidate_relation_scores
        ), breakpoint()

        for relation, label, score in zip(
            candidate_relation_list,
            candidate_relation_labels,
            candidate_relation_scores,
        ):
            relation = (doc_id, tuple(relation))
            if relation not in self._candidate_labels:
                self._candidate_labels[relation] = label

            assert self._candidate_labels[relation] == label

            if (
                relation not in self._candidate_scores
                or self._candidate_scores[relation] < score
            ):
                self._candidate_scores[relation] = score

    @overrides
    def get_metric(self, reset=False):
        if len(self._candidate_scores) == 0:
            return {}

        prediction_scores, gold = [], []
        for k in self._candidate_labels:
            prediction_scores.append(self._candidate_scores[k])
            gold.append(self._candidate_labels[k])

        try :
            threshold = compute_threshold(prediction_scores, gold)
        except :
            breakpoint()
        prediction = [
            1 if self._candidate_scores[k] > threshold else 0
            for k in self._candidate_labels
        ]

        try:
            metrics = pd.io.json.json_normalize(
                classification_report(gold, prediction, output_dict=True), sep="."
            ).to_dict(orient="records")[0]
        except:
            breakpoint()
        metrics = {k.replace(" ", "-"): v for k, v in metrics.items()}

        metrics["1.support_pred"] = sum(prediction)
        metrics["0.support_pred"] = len(prediction) - sum(prediction)

        metrics["threshold"] = float(threshold)

        # Reset counts if at end of epoch.
        if reset:
            self.reset()

        return metrics

    @overrides
    def reset(self):
        self._candidate_labels = {}
        self._candidate_scores = {}
