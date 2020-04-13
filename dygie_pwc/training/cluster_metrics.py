from typing import Any, Dict, List, Tuple
from collections import Counter

from overrides import overrides
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch

from allennlp.training.metrics.metric import Metric


@Metric.register("cluster_coref_scores")
class ConllCorefScores(Metric):
    def __init__(self) -> None:
        self.scorers = [Scorer(m) for m in (Scorer.muc, Scorer.b_cubed, Scorer.ceafe)]

    @overrides
    def __call__(
        self,  # type: ignore
        predicted_clusters_list: List[List[Tuple[int, int]]],
        gold_clusters_list: List[List[Tuple[int, int]]],
    ):
        for predicted_clusters, gold_clusters in zip(predicted_clusters_list, gold_clusters_list):
            predicted_clusters, mention_to_predicted = self.get_gold_clusters(predicted_clusters)
            gold_clusters, mention_to_gold = self.get_gold_clusters(gold_clusters)

            for scorer in self.scorers:
                scorer.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)

    @overrides
    def get_metric(self, reset: bool = False) -> Tuple[float, float, float]:
        metrics = (lambda e: e.get_precision(), lambda e: e.get_recall(), lambda e: e.get_f1())
        precision, recall, f1_score = tuple(
            sum(metric(e) for e in self.scorers) / len(self.scorers) for metric in metrics
        )
        if reset:
            self.reset()
        return precision, recall, f1_score

    @overrides
    def reset(self):
        self.scorers = [Scorer(metric) for metric in (Scorer.muc, Scorer.b_cubed, Scorer.ceafe)]

    @staticmethod
    def get_gold_clusters(gold_clusters):
        gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
        mention_to_gold = {}
        for gold_cluster in gold_clusters:
            for mention in gold_cluster:
                mention_to_gold[mention] = gold_cluster
        return gold_clusters, mention_to_gold


class Scorer:
    """
    Mostly borrowed from <https://github.com/clarkkev/deep-coref/blob/master/evaluation.py>
    """

    def __init__(self, metric):
        self.precision_numerator = 0
        self.precision_denominator = 0
        self.recall_numerator = 0
        self.recall_denominator = 0
        self.metric = metric

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        if self.metric == self.ceafe:
            p_num, p_den, r_num, r_den = self.metric(predicted, gold)
        else:
            p_num, p_den = self.metric(predicted, mention_to_gold)
            r_num, r_den = self.metric(gold, mention_to_predicted)
        self.precision_numerator += p_num
        self.precision_denominator += p_den
        self.recall_numerator += r_num
        self.recall_denominator += r_den

    def get_f1(self):
        precision = (
            0 if self.precision_denominator == 0 else self.precision_numerator / float(self.precision_denominator)
        )
        recall = 0 if self.recall_denominator == 0 else self.recall_numerator / float(self.recall_denominator)
        return 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    def get_recall(self):
        if self.recall_numerator == 0:
            return 0
        else:
            return self.recall_numerator / float(self.recall_denominator)

    def get_precision(self):
        if self.precision_numerator == 0:
            return 0
        else:
            return self.precision_numerator / float(self.precision_denominator)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    @staticmethod
    def b_cubed(clusters, mention_to_gold):
        """
        Averaged per-mention precision and recall.
        <https://pdfs.semanticscholar.org/cfe3/c24695f1c14b78a5b8e95bcbd1c666140fd1.pdf>
        """
        numerator, denominator = 0, 0
        for cluster in clusters:
            if len(cluster) == 1:
                continue
            gold_counts = Counter()
            correct = 0
            for mention in cluster:
                if mention in mention_to_gold:
                    gold_counts[tuple(mention_to_gold[mention])] += 1
            for cluster2, count in gold_counts.items():
                if len(cluster2) != 1:
                    correct += count * count
            numerator += correct / float(len(cluster))
            denominator += len(cluster)
        return numerator, denominator

    @staticmethod
    def muc(clusters, mention_to_gold):
        """
        Counts the mentions in each predicted cluster which need to be re-allocated in
        order for each predicted cluster to be contained by the respective gold cluster.
        <https://aclweb.org/anthology/M/M95/M95-1005.pdf>
        """
        true_p, all_p = 0, 0
        for cluster in clusters:
            all_p += len(cluster) - 1
            true_p += len(cluster)
            linked = set()
            for mention in cluster:
                if mention in mention_to_gold:
                    linked.add(mention_to_gold[mention])
                else:
                    true_p -= 1
            true_p -= len(linked)
        return true_p, all_p

    @staticmethod
    def phi4(gold_clustering, predicted_clustering):
        """
        Subroutine for ceafe. Computes the mention F measure between gold and
        predicted mentions in a cluster.
        """
        return (
            2
            * len([mention for mention in gold_clustering if mention in predicted_clustering])
            / float(len(gold_clustering) + len(predicted_clustering))
        )

    @staticmethod
    def ceafe(clusters, gold_clusters):
        """
        Computes the  Constrained EntityAlignment F-Measure (CEAF) for evaluating coreference.
        Gold and predicted mentions are aligned into clusterings which maximise a metric - in
        this case, the F measure between gold and predicted clusters.

        <https://www.semanticscholar.org/paper/On-Coreference-Resolution-Performance-Metrics-Luo/de133c1f22d0dfe12539e25dda70f28672459b99>
        """
        clusters = [cluster for cluster in clusters if len(cluster) != 1]
        scores = np.zeros((len(gold_clusters), len(clusters)))
        for i, gold_cluster in enumerate(gold_clusters):
            for j, cluster in enumerate(clusters):
                scores[i, j] = Scorer.phi4(gold_cluster, cluster)
        row, col = linear_sum_assignment(-scores)
        similarity = sum(scores[row, col])
        return similarity, len(clusters), similarity, len(gold_clusters)


def match_predicted_clusters_to_gold(
    predicted_clusters_list: List[List[Tuple[int, int]]], gold_clusters_list: List[List[Tuple[int, int]]]
):
    matched_biject = []
    predicted_biject = []
    gold_biject = []
    matched_overlap_p = []
    matched_overlap_g = []

    intersection_scores_list = []

    for predicted_clusters, gold_clusters in zip(predicted_clusters_list, gold_clusters_list):
        predicted_clusters = {i:set(p) for i, p in predicted_clusters.items()}
        gold_clusters = {j:set(g) for j, g in gold_clusters.items()}
        iou_scores = [
            [len(p & g) / len(p | g) for j, g in gold_clusters.items()] for i, p in predicted_clusters.items()
        ]

        intersection_scores = [
            [len(p & g) / len(p) for j, g in gold_clusters.items()] for i, p in predicted_clusters.items()
        ]

        intersection_scores_dict = {}
        for i, k in enumerate(predicted_clusters) :
            intersection_scores_dict[k] = {}
            for j, l in enumerate(gold_clusters) :
                if intersection_scores[i][j] > 0 :
                    intersection_scores_dict[k][l] = intersection_scores[i][j]

        intersection_scores_list.append(intersection_scores_dict)

        if len(predicted_clusters) == 0:
            matched_biject.append(0)
            predicted_biject.append(0)
            gold_biject.append(len(gold_clusters))
            matched_overlap_p.append(0)
            matched_overlap_g.append(0)
            continue

        p_match_g = np.where(np.array(iou_scores) > 0.5, 1, 0)
        p_overlap_g = np.where(np.array(intersection_scores) > 0.5, 1, 0)

        matched_biject.append(p_match_g.sum())
        predicted_biject.append(p_match_g.shape[0])
        gold_biject.append(p_match_g.shape[1])

        matched_overlap_p.append((p_overlap_g.sum(-1) > 0).sum())
        matched_overlap_g.append((p_overlap_g.sum(0) > 0).sum())

    macro_p = np.nan_to_num(np.array(matched_biject) / np.array(predicted_biject))
    macro_r = np.nan_to_num(np.array(matched_biject) / np.array(gold_biject))

    metrics = {
        "macro_p": np.mean(macro_p),
        "macro_r": np.mean(macro_r),
        "macro_overlap_p": np.mean(np.nan_to_num(np.array(matched_overlap_p) / np.array(predicted_biject))),
        "macro_overlap_r": np.mean(np.nan_to_num(np.array(matched_overlap_g) / np.array(gold_biject)))
    }

    metrics["macro_f1"] = (2 * metrics["macro_p"] * metrics["macro_r"]) / (metrics["macro_p"] + metrics["macro_r"])
    metrics["macro_overlap_f1"] = (2 * metrics["macro_overlap_p"] * metrics["macro_overlap_r"]) / (metrics["macro_overlap_p"] + metrics["macro_overlap_r"])


    return metrics, intersection_scores_list