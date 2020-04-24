#! /usr/bin/env python
from typing import Dict, List, Tuple

from tqdm import tqdm

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--gold-file")
parser.add_argument("--cluster-file")
parser.add_argument("--output-file")

from scirex.predictors.utils import *
from scirex_utilities.convert_brat_annotations_to_json import load_jsonl

import logging

logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s", level=logging.INFO)


def predict(clusters_file, gold_file, output_file):
    gold_data = {item["doc_id"]: item for item in load_jsonl(gold_file)}
    for item in gold_data.values() :
        merge_method_subrelations(item)
        
    clusters_data = load_jsonl(clusters_file)

    with open(output_file, "w") as f:
        for doc in clusters_data:
            gold_doc = gold_data[doc["doc_id"]]
            gold_spans: List[tuple] = convert_ner_to_list(gold_doc["ner"])
            predicted_spans: List[tuple] = convert_ner_to_list(doc["spans"])

            predicted_to_gold_map: Dict[tuple, tuple] = map_predicted_spans_to_gold(
                predicted_spans, gold_spans
            )

            intersection_scores = map_and_intersect_predicted_clusters_to_gold(
                predicted_clusters=doc["clusters"],
                gold_clusters=gold_doc["coref"],
                predicted_to_gold_map=predicted_to_gold_map,
            )

            intersection_scores = {
                k: max([(g, v) for g, v in gold_c.items()], key=lambda x: x[1])[0]
                for k, gold_c in intersection_scores.items()
                if len(gold_c) > 0
            }

            salient_clusters = {k: v for k, v in doc["clusters"].items() if k in intersection_scores and len(v) > 0}

            f.write(
                json.dumps({"doc_id": doc["doc_id"], "clusters": salient_clusters, "spans": doc["spans"]})
                + "\n"
            )


def main(args):
    predict(args.cluster_file, args.gold_file, args.output_file)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
