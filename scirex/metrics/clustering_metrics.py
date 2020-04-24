from typing import Any, Dict, List, Tuple

from scirex.predictors.utils import map_and_intersect_predicted_clusters_to_gold


def match_predicted_clusters_to_gold(
    predicted_clusters: Dict[str, List[Tuple[int, int]]],
    gold_clusters: Dict[str, List[Tuple[int, int]]],
    span_map,
    words
):
    intersection_scores = map_and_intersect_predicted_clusters_to_gold(predicted_clusters, gold_clusters, span_map)
    matched_clusters = {}
    
    for p in intersection_scores :
        if len(intersection_scores[p]) > 0:
            g, v = max(list(intersection_scores[p].items()), key=lambda x : x[1])
            if v > 0.5 :
                matched_clusters[p] = g

    metrics = {'p' : len(matched_clusters) / (len(predicted_clusters) + 1e-7), 'r' : len(set(matched_clusters.values())) / (len(gold_clusters) + 1e-7)}
    metrics['f1'] = 2 * metrics['p'] * metrics['r'] / (metrics['p'] + metrics['r'] + 1e-7)

    return metrics, matched_clusters
