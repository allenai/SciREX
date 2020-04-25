import argparse
from itertools import combinations, product
from typing import Dict

import pandas as pd

from scirex.metrics.clustering_metrics import match_predicted_clusters_to_gold
from scirex.predictors.utils import map_predicted_spans_to_gold, merge_method_subrelations
from scirex_utilities.entity_utils import used_entities
from scirex_utilities.json_utilities import load_jsonl

parser = argparse.ArgumentParser()
parser.add_argument("--gold-file")
parser.add_argument("--ner-file")
parser.add_argument("--clusters-file")
parser.add_argument("--relations-file")

def has_all_mentions(doc, relation):
    has_mentions = True
    for e in used_entities:
        if e != "Method":
            has_mentions = has_mentions and (len(doc["clusters"][relation[e]]) > 0)
        else:
            has_mentions = has_mentions and (len(doc["clusters"][relation[e]]) > 0)

    return has_mentions


def convert_to_dict(data):
    return {x["doc_id"]: x for x in data}


def ner_metrics(gold_data, predicted_data):
    mapping = {}
    for doc in gold_data:
        predicted_doc = predicted_data[doc["doc_id"]]
        predicted_spans = predicted_doc["ner"]
        gold_spans = doc["ner"]

        mapping[doc["doc_id"]] = map_predicted_spans_to_gold(predicted_spans, gold_spans)

    return mapping


def clustering_metrics(gold_data, predicted_clusters, span_map):
    all_metrics = []
    mappings = {}
    for doc in gold_data:
        predicted_doc = predicted_clusters[doc["doc_id"]]
        metrics, mapping = match_predicted_clusters_to_gold(
            predicted_doc["clusters"], doc["coref"], span_map[doc["doc_id"]], doc['words']
        )
        mappings[doc["doc_id"]] = mapping
        all_metrics.append(metrics)

    all_metrics = pd.DataFrame(all_metrics)
    print("Salient Clustering Metrics")
    print(all_metrics.describe().loc['mean'])

    return mappings


def get_types_of_clusters(predicted_ner, predicted_clusters):
    for doc_id in predicted_clusters:
        clusters = predicted_clusters[doc_id]["clusters"]
        ner = {(x[0], x[1]): x[2] for x in predicted_ner[doc_id]["ner"]}

        predicted_clusters[doc_id]["types"] = {}
        for c, spans in clusters.items():
            types = set([ner[tuple(span)] for span in spans])
            if len(types) == 0:
                predicted_clusters[doc_id]["types"][c] = "Empty"
                continue
            predicted_clusters[doc_id]["types"][c] = list(types)[0]


def are_they_same(pr, gr, types=used_entities):
    same = True
    for e in types:
        if e != "Method":
            same = same and (pr[e] == gr[e])
        else:
            same = same and (pr[e] == gr[e])

    return 1 if same else 0


def main(args):
    gold_data = load_jsonl(args.gold_file)
    for d in gold_data:
        merge_method_subrelations(d)
        d["clusters"] = d["coref"]

    predicted_ner = convert_to_dict(load_jsonl(args.ner_file))
    predicted_salient_clusters = convert_to_dict(load_jsonl(args.clusters_file))
    for d, doc in predicted_salient_clusters.items() :
        if 'clusters' not in doc :
            merge_method_subrelations(doc)
            doc['clusters'] = {x:v for x, v in doc['coref'].items() if len(v) > 0}

    predicted_relations = convert_to_dict(load_jsonl(args.relations_file))

    predicted_span_to_gold_span_map: Dict[str, Dict[tuple, tuple]] = ner_metrics(gold_data, predicted_ner)
    get_types_of_clusters(predicted_ner, predicted_salient_clusters)
    get_types_of_clusters(convert_to_dict(gold_data), convert_to_dict(gold_data))
    predicted_cluster_to_gold_cluster_map = clustering_metrics(
        gold_data, predicted_salient_clusters, predicted_span_to_gold_span_map
    )

    for n in [2, 4] :
        all_metrics = []
        for types in combinations(used_entities, n):
            for doc in gold_data:
                predicted_data = predicted_relations[doc["doc_id"]]
                mapping = predicted_cluster_to_gold_cluster_map[doc["doc_id"]]

                relations = list(set([
                    tuple([mapping.get(v, v) for v in x[0]])
                    for x in predicted_data["predicted_relations"]
                    if x[2] == 1
                ]))

                relations = [dict(zip(used_entities, x)) for x in relations]

                gold_relations = [x for x in doc['n_ary_relations'] if has_all_mentions(doc, x)]

                matched_predicted = []
                matched_gold = []
                for pr, gr in product(relations, gold_relations):
                    if are_they_same(pr, gr, types):
                        if gr not in matched_gold:
                            matched_gold.append(gr)
                        if pr not in matched_predicted:
                            matched_predicted.append(pr)

                metrics = {
                    "p": len(matched_predicted) / (len(relations) + 1e-7),
                    "r": len(matched_gold) / (len(gold_relations) + 1e-7),
                }
                metrics["f1"] = 2 * metrics["p"] * metrics["r"] / (metrics["p"] + metrics["r"] + 1e-7)

                if len(gold_relations) > 0:
                    all_metrics.append(metrics)

        all_metrics = pd.DataFrame(all_metrics)
        print(f"Relation Metrics n={n}")
        print(all_metrics.describe().loc['mean'])


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
