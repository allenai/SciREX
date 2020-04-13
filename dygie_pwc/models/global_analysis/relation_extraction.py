import json
import numpy as np
from scripts.entity_utils import *
from itertools import combinations, product
from dygie_pwc.models.global_analysis import *

from dygie_pwc.data.dataset_readers.read_pwc_dataset import is_x_in_y


def get_relations_between_clusters(document, span_field, cluster_labels):
    relation = np.zeros((max(cluster_labels) + 1, max(cluster_labels) + 1))
    idx2span = {i: tuple(k) for i, k in enumerate(document[span_field])}
    relation_matrix = generate_matrix_for_document(document, span_field, "relation_scores")
    paragraphs = document["paragraphs"]
    map_span_to_paragraphs = {}
    for span in document[span_field]:
        for i, p in enumerate(paragraphs):
            if is_x_in_y(span, p):
                map_span_to_paragraphs[span] = i

    relation_matrix = (relation_matrix + relation_matrix.T) / 2
    for i in range(len(idx2span)):
        for j in range(len(idx2span)):
            dist = map_span_to_paragraphs[idx2span[i]] - map_span_to_paragraphs[idx2span[j]]
            if i != j:
                relation[cluster_labels[i], cluster_labels[j]] = max(
                    relation[cluster_labels[i], cluster_labels[j]], relation_matrix[i, j]
                )

    return relation


def generate_relations(clusters, relation_matrix, linked_clusters, n, threshold):
    cluster_type_list = {k: [] for k in used_entities}
    for i in linked_clusters:
        cluster_type_list[clusters[i]["type"]].append(i)

    outputs = {}
    for type_list in combinations(used_entities, n):
        outputs[type_list] = []
        cluster_lists = [cluster_type_list[k] for k in type_list]
        for clist in product(*cluster_lists):
            all_related = 0.0
            c = 0
            for c1, c2 in combinations(clist, 2):
                all_related += relation_matrix[c1, c2]
                c += 1
            if c == 0 or all_related >= threshold:
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
            pred_rellist = [tuple([clusters[i]["matched"] for i in rel]) for rel in rellist]
            pred_rellist = [x for x in pred_rellist if len(x) == n]
            true_rellist = true_relation[n][t]

            scores[n][t]["matched"] += len(set(pred_rellist) & set(true_rellist))
            scores[n][t]["predicted"] += len(set(pred_rellist))
            scores[n][t]["gold"] += len(set(true_rellist))

            p, r, f1 = compute_prf_n(
                len(set(pred_rellist)), len(set(true_rellist)), len(set(pred_rellist) & set(true_rellist))
            )
            scores[n][t]["macro_p"] += p
            scores[n][t]["macro_r"] += r
            scores[n][t]["macro_f1"] += f1


def scores_scaffold():
    return {
        n: {
            type_list: {"matched": 0, "predicted": 0, "gold": 0, "macro_p": 0.0, "macro_r": 0.0, "macro_f1": 0.0}
            for type_list in combinations(used_entities, n)
        }
        for n in range(1, 6)
    }


def run_eval_for_relation_extraction(documents):

    scores_gold = scores_scaffold()
    scores_predicted = scores_scaffold()
    scores_oracle = scores_scaffold()
    scores_oracle_expert = scores_scaffold()
    scores_oracle_expert_table = scores_scaffold()
    scores_oracle_abstract = scores_scaffold()

    for d in documents:
        oracle_clusters = {k: {"matched": k if len(v["spans"]) > 0 else None} for k, v in d["gold_clusters"].items()}
        evaluation_relation_extraction(d["gold_clusters"], d["true_relations"], d["true_relations"], scores_gold)
        evaluation_relation_extraction(oracle_clusters, d["true_relations"], d["true_relations"], scores_oracle)
        evaluation_relation_extraction(d["clusters"], d["predicted_relations"], d["true_relations"], scores_predicted)

        expert_oracle_clusters = {
            k: {"matched": k if (len(v["spans"]) > 0 or k in d["missed"]) else None}
            for k, v in d["gold_clusters"].items()
        }

        evaluation_relation_extraction(
            expert_oracle_clusters, d["true_relations"], d["true_relations"], scores_oracle_expert
        )

        expert_oracle_clusters = {
            k: {"matched": k if (len(v["spans"]) > 0 or k in d["missed"] or k in d["table"]) else None}
            for k, v in d["gold_clusters"].items()
        }

        evaluation_relation_extraction(
            expert_oracle_clusters, d["true_relations"], d["true_relations"], scores_oracle_expert_table
        )

        # abstract_oracle_clusters = {
        #     k: {"matched": k if (len([x for x in v["spans"] if x[1] < 400]) > 0) else None}
        #     for k, v in d["gold_clusters"].items()
        # }

        # abstract_relations = {
        #     n: {
        #         t: [rel for rel in rellist if all([abstract_oracle_clusters[x]["matched"] != None for x in rel])]
        #         for t, rellist in reltypes.items()
        #     }
        #     for n, reltypes in d["true_relations"].items()
        # }

        # evaluation_relation_extraction(
        #     abstract_oracle_clusters, abstrac, d["true_relations"], scores_oracle_abstract
        # )

    scores_gold = prf_all(scores_gold, len(documents))
    scores_oracle = prf_all(scores_oracle, len(documents))
    scores_predicted = prf_all(scores_predicted, len(documents))
    scores_oracle_expert = prf_all(scores_oracle_expert, len(documents))
    scores_oracle_expert_table = prf_all(scores_oracle_expert_table, len(documents))
    return scores_oracle, scores_oracle_expert, scores_oracle_expert_table, scores_gold, scores_predicted


def prf_all(scores, n_d):
    return {k: {",".join(x): compute_prf(r, n_d) for x, r in v.items()} for k, v in scores.items()}


def compute_prf(d, n_d):
    nd = {}
    nd["p"] = safe_div(d["matched"], d["predicted"])
    nd["r"] = safe_div(d["matched"], d["gold"])
    nd["f1"] = safe_div((2 * nd["p"] * nd["r"]), (nd["p"] + nd["r"]))
    nd.update(d)
    nd["macro_p"] /= n_d
    nd["macro_r"] /= n_d
    nd["macro_f1"] /= n_d
    return nd


def compute_prf_n(predicted, gold, matched):
    p = safe_div(matched, predicted)
    r = safe_div(matched, gold)
    f1 = safe_div((2 * p * r), (p + r))
    return p, r, f1


def safe_div(a, b):
    if b != 0:
        return a / b
    else:
        return 0.0

