#! /usr/bin/env python
from collections import defaultdict
import json
import os
from sys import argv
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gold-file")
parser.add_argument("--cluster-file")

from scirex.commands.utils import *
from scirex_utilities.convert_brat_annotations_to_json import load_jsonl, annotations_to_jsonl

import logging

logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s", level=logging.INFO)


def predict(gold_file, predicted_cluster_file):
    gold_data = load_jsonl(gold_file)
    predicted_data = load_jsonl(predicted_cluster_file)

    for gold_doc, predicted_doc in tqdm(zip(gold_data, predicted_data)):
        assert gold_doc["doc_id"] == predicted_doc["doc_id"]
        gold_doc["ner"] = convert_ner_to_dict(gold_doc["ner"])
        predicted_doc["ner"] = convert_ner_to_dict(predicted_doc["ner"])

        predicted_to_gold_map = map_predicted_spans_to_gold(predicted_doc["ner"], gold_doc["ner"])

        intersection_scores = map_and_intersect_predicted_clusters_to_gold(
            predicted_clusters=predicted_doc["coref"],
            gold_clusters=gold_doc["coref"],
            predicted_to_gold_map=predicted_to_gold_map,
        )

        intersection_scores = {
            k: max([(g, v) for g, v in gold_c.items()], key=lambda x: x[1])[0]
            for k, gold_c in intersection_scores.items()
            if len(gold_c) > 0
        }

        predicted_doc['coref'] = {k:v for k, v in predicted_doc['coref'].items() if k in intersection_scores}
        predicted_doc['ner'] = convert_ner_to_list(predicted_doc['ner'])

    annotations_to_jsonl(predicted_data, predicted_cluster_file + '.linked_from_gold')


def main(args):
    test_file = args.gold_file
    output_file = args.cluster_file
    predict(test_file, output_file)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
