import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from scirex.metrics.f1 import compute_f1
from scirex.commands.utils import map_predicted_spans_to_gold, span_match
from itertools import product, combinations

from scirex_utilities.entity_utils import used_entities

parser = argparse.ArgumentParser()
parser.add_argument("--test-file")
parser.add_argument("--output-dir")
parser.add_argument("--gold", action="store_true")


def load_jsonl(filename):
    with open(filename) as f:
        data = [json.loads(line) for line in f]

    return data

def load_documents(test_file) :
    documents = load_jsonl(test_file)
    documents = {x["doc_id"]: x for x in documents}

    for d in documents :
        coref_non_empty = {k:v for k, v in documents[d]['coref'].items() if len(v) > 0}
        method_subrelations = {k:set([x[1] for x in v]) for k, v in documents[d]['method_subrelations'].items()}

        filtered_relations = []
        for r in documents[d]['n_ary_relations'] :
            all_there = True
            for k, v in r.items() :
                if k == 'score' :
                    continue
                if k == 'Method' :
                    all_there &= v in coref_non_empty or any(sub in coref_non_empty for sub in method_subrelations[v])
                else :
                    all_there &= v in coref_non_empty

            if all_there :
                filtered_relations.append(r)

        documents[d]['coref'] = coref_non_empty
        documents[d]['n_ary_relations'] = filtered_relations

    return documents

def update_documents_with_spans(documents, span_file, args) :
    spans = load_jsonl(span_file)

    for doc in spans:
        documents[doc["doc_id"]]["predicted_spans"] = doc["prediction"]# if not args.gold else "gold"]

    for d in documents.values():
        d["predicted_spans"] = {tuple(x["span"]): x["label"] for x in d["predicted_spans"]}
        d["gold_spans"] = {(x[0], x[1]): x[2] for x in d["ner"]}

    generate_metrics_for_ner(documents)

    return documents

def update_documents_with_clusters(documents, cluster_file) :
    clusters = load_jsonl(cluster_file)
    clusters_links = load_jsonl(cluster_file + '.linked')
    clusters_links = {x['doc_id']: x for x in clusters_links}
    for doc in clusters:
        documents[doc["doc_id"]]["predicted_clusters"] = doc["coref"]
        documents[doc["doc_id"]]["linked_clusters"] = [x for x, v in clusters_links[doc["doc_id"]]['coref'].items()]
        documents[doc["doc_id"]]["gold_clusters"] = doc["coref"]

    intersection_scores_list = generate_metrics_for_cluster(documents)
    for i, d in enumerate(documents.keys()) :
        documents[d]['intersection_scores'] = intersection_scores_list[i]
        
    return documents

def main(args):
    output_folder = args.output_dir
    test_file = args.test_file
    span_file = os.path.join(output_folder, "spans.jsonl")
    cluster_file = os.path.join(output_folder, "clusters.jsonl" + (".linked_from_gold" if args.gold else ""))
    n_ary_relations_file = os.path.join(output_folder, "n_ary_relations.jsonl" + (".linked_from_gold" if args.gold else ""))

    documents = load_documents(test_file)
    documents = update_documents_with_spans(documents, span_file, args)
    documents = update_documents_with_clusters(documents, cluster_file)

    relations = load_jsonl(n_ary_relations_file)
    for doc in relations:
        predicted_relations = [tuple(x) for x in doc["predicted_relations"]]
        scores = doc["scores"]
        collapse_relations = {}
        for rel, score in zip(predicted_relations, scores):
            if rel in collapse_relations:
                collapse_relations[rel] = max(collapse_relations[rel], score)
            else:
                collapse_relations[rel] = score

        documents[doc["doc_id"]]["predicted_n_ary_relations"] = collapse_relations

    intersection_scores_list = generate_metrics_for_cluster(documents)
    precision = defaultdict(list)
    recall = defaultdict(list)

    for i, d in enumerate(documents):
        predicted_to_gold_list = intersection_scores_list[i]
        predicted_to_gold_list = {
            p: max([(g, v) for g, v in gold_c.items()] + [(p, 0.0)], key=lambda x: x[1])
            for p, gold_c in predicted_to_gold_list.items()
        }

        predicted_relations = documents[d]["predicted_n_ary_relations"]
        gold_relations = documents[d]["n_ary_relations"]
        method_subrelations = documents[d]["method_subrelations"]

        if len(gold_relations) == 0 :
            continue

        invert_subrelations = defaultdict(list)
        for m, ms in method_subrelations.items():
            for span, n in ms:
                invert_subrelations[n].append(m)

        mapped_relations = defaultdict(float)
        for relation, score in predicted_relations.items():
            new_relation = []
            for r in relation:
                m = predicted_to_gold_list[r][0]
                if m in invert_subrelations:
                    new_relation.append(invert_subrelations[m])
                else:
                    new_relation.append([m])

            new_relations = list(product(*new_relation))
            for r in new_relations:
                mapped_relations[tuple(r)] = max(mapped_relations[tuple(r)], score)

        predicted_relations = set([tuple(k) for k, v in mapped_relations.items() if v > 0.777])
        gold_relations = set([tuple([r[e] for e in used_entities]) for r in gold_relations])

        for n_ary in range(1, 5):
            for elist in combinations(used_entities, n_ary):
                prel = set(
                    [tuple([r for i, r in enumerate(p) if used_entities[i] in elist]) for p in predicted_relations]
                )
                grel = set([tuple([r for i, r in enumerate(p) if used_entities[i] in elist]) for p in gold_relations])

                precision[" / ".join(elist)].append(len(prel & grel) / len(prel) if len(prel) > 0 else 0)
                recall[" / ".join(elist)].append(len(prel & grel) / len(grel) if len(grel) > 0 else 0)


    metrics = {}
    for label in precision:
        p = np.mean(precision[label])
        r = np.mean(recall[label])
        metrics[label] = {
            "P": p,
            "R": r,
            "F1": (2 * p * r) / (p + r),
        }

    metrics = pd.DataFrame(metrics).T
    print(metrics)

    metrics['n_ary'] = pd.Series(metrics.index).apply(lambda x : len(x.split('/'))).values

    print(metrics.groupby('n_ary').agg(np.mean).to_latex(float_format=lambda x: "{:0.3f}".format(x)))


def generate_metrics_for_ner_single_document(predicted_spans, gold_spans):
    typed_predicted_spans = defaultdict(list)
    typed_gold_spans = defaultdict(list)

    for span, label in predicted_spans.items():
        typed_predicted_spans[label].append(span)

    for span, label in gold_spans.items():
        typed_gold_spans[label].append(span)

    total_predicted = {}
    total_gold = {}
    total_soft_matched = {}
    total_exact_matched = {}

    for label in typed_gold_spans.keys():
        predicted = typed_predicted_spans[label]
        gold = typed_gold_spans[label]
        exact_match = np.array([[1 if p == g else 0 for g in gold] for p in predicted])
        iou_scores = np.array([[1 if span_match(p, g) > 0.5 else 0 for g in gold] for p in predicted])

        total_gold[label] = len(gold)
        total_predicted[label] = len(predicted)
        total_exact_matched[label] = exact_match.sum()
        total_soft_matched[label] = iou_scores.sum()

    return total_exact_matched, total_soft_matched, total_gold, total_predicted


def generate_metrics_for_ner(documents):
    macro_exact_p, macro_exact_r = defaultdict(int), defaultdict(int)
    macro_soft_p, macro_soft_r = defaultdict(int), defaultdict(int)

    for d in documents.values():
        doc_em, doc_sm, doc_g, doc_p = generate_metrics_for_ner_single_document(d["predicted_spans"], d["gold_spans"])
        for label in doc_sm:
            doc_exact_p, doc_exact_r, doc_exact_f1 = compute_f1(doc_p[label], doc_g[label], doc_em[label], m=1)
            doc_soft_p, doc_soft_r, doc_soft_f1 = compute_f1(doc_p[label], doc_g[label], doc_sm[label], m=1)

            macro_exact_p[label] += doc_exact_p / len(documents)
            macro_exact_r[label] += doc_exact_r / len(documents)

            macro_soft_p[label] += doc_soft_p / len(documents)
            macro_soft_r[label] += doc_soft_r / len(documents)

    metrics = {}
    for label in macro_exact_p.keys():
        metrics[label] = {}
        metrics[label]["exact_p"] = macro_exact_p[label]
        metrics[label]["exact_r"] = macro_exact_r[label]
        metrics[label]["exact_f1"] = (2 * macro_exact_p[label] * macro_exact_r[label]) / (
            macro_exact_p[label] + macro_exact_r[label]
        )

        metrics[label]["soft_p"] = macro_soft_p[label]
        metrics[label]["soft_r"] = macro_soft_r[label]
        metrics[label]["soft_f1"] = (2 * macro_soft_p[label] * macro_soft_r[label]) / (
            macro_soft_p[label] + macro_soft_r[label]
        )

    print(
        pd.DataFrame(metrics)
        .T.rename(columns=lambda x: x.replace("_", " ").title())
        .to_latex(float_format=lambda x: "{:0.3f}".format(x))
    )

    return metrics


from scirex.metrics.cluster_metrics import match_predicted_clusters_to_gold


def generate_metrics_for_cluster(documents):
    predicted_clusters_list = []
    gold_clusters_list = []

    for d in documents.values():
        predicted_to_gold = map_predicted_spans_to_gold(d["predicted_spans"], d["gold_spans"])
        for cluster in d["predicted_clusters"].values():
            for span in cluster:
                assert tuple(span) in d["predicted_spans"], breakpoint()

        predicted_clusters_list.append(
            {
                k: [predicted_to_gold[tuple(span)] for span in p]
                for k, p in d["predicted_clusters"].items()
                if k in d["linked_clusters"]
            }
        )
        gold_clusters_list.append({k: [tuple(span) for span in cluster] for k, cluster in d["coref"].items()})

    metrics, intersection_scores_list = match_predicted_clusters_to_gold(predicted_clusters_list, gold_clusters_list)
    print(pd.Series(metrics))

    return intersection_scores_list


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
