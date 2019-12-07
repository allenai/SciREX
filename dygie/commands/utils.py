from collections import defaultdict


def span_match(span_1, span_2):
    sa, ea = span_1
    sb, eb = span_2
    iou = (min(ea, eb) - max(sa, sb)) / (max(eb, ea) - min(sa, sb))
    return iou


def map_predicted_spans_to_gold(predicted_spans, gold_spans):
    typed_predicted_spans = defaultdict(list)
    typed_gold_spans = defaultdict(list)

    for span, label in predicted_spans.items():
        typed_predicted_spans[label].append(span)

    for span, label in gold_spans.items():
        typed_gold_spans[label].append(span)

    predicted_to_gold = {}

    for p, label in predicted_spans.items():
        gold = typed_gold_spans[label]
        predicted_to_gold[p] = p
        for g in gold:
            if span_match(p, g) > 0.5:
                predicted_to_gold[p] = g
                break

    for p in predicted_spans:
        assert p in predicted_to_gold, breakpoint()

    return predicted_to_gold


from typing import Dict, List, Tuple


def intersect_predicted_clusters_to_gold(
    predicted_clusters: Dict[str, List[Tuple[int, int]]], gold_clusters: Dict[str, List[Tuple[int, int]]]
):

    predicted_clusters = {i: set(p) for i, p in predicted_clusters.items()}
    gold_clusters = {j: set(g) for j, g in gold_clusters.items()}

    intersection_scores = [
        [len(p & g) / len(p) for j, g in gold_clusters.items()] for i, p in predicted_clusters.items()
    ]

    intersection_scores_dict = {}
    for i, k in enumerate(predicted_clusters):
        intersection_scores_dict[k] = {}
        for j, l in enumerate(gold_clusters):
            if intersection_scores[i][j] > 0:
                intersection_scores_dict[k][l] = intersection_scores[i][j]


    return intersection_scores_dict


def map_and_intersect_predicted_clusters_to_gold(
    predicted_clusters: Dict[str, List[Tuple[int, int]]],
    gold_clusters: Dict[str, List[Tuple[int, int]]],
    predicted_to_gold_map,
):

    predicted_clusters = {k:[predicted_to_gold_map[tuple(x)] for x in v] for k, v in predicted_clusters.items()}
    gold_clusters = {k:[tuple(x) for x in v] for k, v in gold_clusters.items()}
    intersection_scores = intersect_predicted_clusters_to_gold(predicted_clusters, gold_clusters)

    return intersection_scores

def convert_ner_to_dict(ner: Tuple[int, int, str]) :
    return {(x[0], x[1]):x[2] for x in ner}

def convert_ner_to_list(ner: Dict[Tuple[int, int], str]) :
    return [(k[0], k[1], v) for k, v in ner.items()]

