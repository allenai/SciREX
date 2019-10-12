from overrides import overrides
from allennlp.training.metrics.metric import Metric
from dygie.training.f1 import compute_f1, safe_div


class NAryRelationMetrics(Metric):
    def __init__(self):
        self.reset()

    @overrides
    def __call__(self, candidate_relation_list, gold_relation_list, candidate_relation_score, doc_id):
        candidate_relation_score, = self.unwrap_to_tensors(candidate_relation_score.squeeze(0))
        candidate_relation_score = list(candidate_relation_score.numpy())
        assert len(candidate_relation_list) == len(candidate_relation_score), breakpoint()
        candidate_relation_list = [(doc_id, tuple(rel)) for rel in candidate_relation_list]
        gold_relation_list = [(doc_id, tuple(rel)) for rel in gold_relation_list]

        self._total_gold |= set(gold_relation_list)
        for cr, sc in zip(candidate_relation_list, candidate_relation_score) :
            if cr not in self._total_candidates or self._total_candidates[cr] < sc :
                self._total_candidates[cr] = sc

    @overrides
    def get_metric(self, threshold, reset=False):
        predicted_relation_list = set([cr for cr, v in self._total_candidates.items() if v > threshold])
        matched_relation_list = predicted_relation_list & self._total_gold

        precision, recall, f1 = compute_f1(len(predicted_relation_list), len(self._total_gold), len(matched_relation_list))

        matched_candidate_ratio = safe_div(len(matched_relation_list), len(self._total_candidates))
        predicted_candidate_ratio = safe_div(len(predicted_relation_list), len(self._total_candidates))

        gold_candidate_overlap = len(self._total_gold & set(self._total_candidates.keys()))
        considered_ratio = safe_div(gold_candidate_overlap, len(self._total_gold))
        gold_ratio = safe_div(gold_candidate_overlap, len(self._total_candidates))

        # Reset counts if at end of epoch.
        if reset:
            self.reset()

        return {
            "precision" : precision,
            "recall" : recall,
            "f1-measure" : f1,
            "matched_candidate_ratio" : matched_candidate_ratio,
            "predicted_candidate_ratio" : predicted_candidate_ratio,
            'considered_ratio' : considered_ratio,
            'gold_ratio' : gold_ratio
        }

    @overrides
    def reset(self):
        self._total_gold = set()
        self._total_candidates = {}