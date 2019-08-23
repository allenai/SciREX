import json
import numpy as np
from scripts.entity_utils import *
from itertools import combinations, product
from dygie.models.global_analysis import *

from dygie.data.dataset_readers.read_pwc_dataset import is_x_in_y


def get_relations_between_clusters(document, span_field, cluster_labels, relation_threshold):
    relation = np.zeros((max(cluster_labels) + 1, max(cluster_labels) + 1))
    idx2span = {i: tuple(k) for i, k in enumerate(document[span_field])}
    relation_matrix = generate_matrix_for_document(document, span_field, "relation_scores")
    paragraphs = document["paragraphs"]
    map_span_to_paragraphs = {}
    for span in document[span_field]:
        for i, p in enumerate(paragraphs):
            if is_x_in_y(span, p):
                map_span_to_paragraphs[span] = i

    relation_matrix = ((relation_matrix + relation_matrix.T) / 2 > relation_threshold).astype(int)
    for i in range(len(idx2span)):
        for j in range(len(idx2span)):
            dist = map_span_to_paragraphs[idx2span[i]] - map_span_to_paragraphs[idx2span[j]]
            if i != j and abs(dist) < 1:
                relation[cluster_labels[i], cluster_labels[j]] += relation_matrix[i, j]

    return relation


def generate_relations(clusters, relation_matrix, linked_clusters, n):
    cluster_type_list = {k: [] for k in used_entities}
    for i in linked_clusters:
        cluster_type_list[clusters[i]["type"]].append(i)

    outputs = {}
    for type_list in combinations(used_entities, n):
        outputs[type_list] = []
        cluster_lists = [cluster_type_list[k] for k in type_list]
        for clist in product(*cluster_lists):
            all_related = True
            for c1, c2 in combinations(clist, 2):
                all_related &= relation_matrix[c1, c2] > 0
            if all_related:
                outputs[type_list].append(clist)

    return outputs


def generate_true_relations(n_ary_relations, n):
    outputs = {}
    for type_list in combinations(used_entities, n):
        outputs[type_list] = set()
        for rel in n_ary_relations:
            tp = tuple(rel[i] for i in type_list)
            if tp not in outputs[type_list]:
                outputs[type_list].add(tp)

        outputs[type_list] = list(outputs[type_list])
    return outputs


def evaluation_relation_extraction(clusters, relations, true_relation, scores):
    for n, reltypes in relations.items():
        for t, rellist in reltypes.items():
            pred_rellist = [tuple([clusters[i]['matched'] for i in rel]) for rel in rellist]
            pred_rellist = [x for x in pred_rellist if len(x) == n]
            true_rellist = true_relation[n][t]
            # assert len(pred_rellist) <= len(true_rellist)
            # assert len(set(pred_rellist) & set(true_rellist)) == len(set(pred_rellist)), breakpoint()
            scores[n][t]["matched"] += len(set(pred_rellist) & set(true_rellist))
            scores[n][t]["predicted"] += len(set(pred_rellist))
            scores[n][t]["gold"] += len(set(true_rellist))
            # if len(t)  == 1 :
            #     print("Evaluation", pred_rellist)
            #     print("Gold", true_rellist)
            #     print("="*20)


def run_eval_for_relation_extraction(documents):
    scores_oracle = {
        n: {type_list: {"matched": 0, "predicted": 0, "gold": 0} for type_list in combinations(used_entities, n)}
        for n in range(1, 6)
    }

    scores_predicted = {
        n: {type_list: {"matched": 0, "predicted": 0, "gold": 0} for type_list in combinations(used_entities, n)}
        for n in range(1, 6)
    }

    for d in documents:
        evaluation_relation_extraction(d["gold_clusters"], d["true_relations"], d["true_relations"], scores_oracle)
        evaluation_relation_extraction(d["clusters"], d["predicted_relations"], d["true_relations"], scores_predicted)

    scores_oracle = {k: {",".join(x): compute_prf(r) for x, r in v.items()} for k, v in scores_oracle.items()}
    scores_predicted = {k: {",".join(x): compute_prf(r) for x, r in v.items()} for k, v in scores_predicted.items()}
    return scores_oracle, scores_predicted


def compute_prf(d):
    nd = {}
    nd["p"] = safe_div(d["matched"], d["predicted"])
    nd["r"] = safe_div(d["matched"], d["gold"])
    nd["f1"] = safe_div((2 * nd["p"] * nd["r"]), (nd["p"] + nd["r"]))
    return nd


def safe_div(a, b):
    if b != 0:
        return a / b
    else:
        return 0.0

