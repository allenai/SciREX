import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from scirex.training.f1 import compute_f1
from scirex.training.span_f1_metrics import span_match
from itertools import product, combinations

from scripts.entity_utils import used_entities

parser = argparse.ArgumentParser()
parser.add_argument("--test-file")
parser.add_argument("--output-dir")
parser.add_argument("--card", type=int)


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

def main(args):
    output_folder = args.output_dir
    test_file = args.test_file
    n_ary_relations_file = os.path.join(output_folder, "n_ary_relations.jsonl.gold_salient_clusters_" + str(args.card))

    documents = load_documents(test_file)
    relations = load_jsonl(n_ary_relations_file)
    
    for doc in relations:
        predicted_relations = [tuple(x) for x in doc["predicted_relations"]]
        scores = doc["is_true"]
        collapse_relations = {}
        for rel, score in zip(predicted_relations, scores):
            if rel in collapse_relations:
                collapse_relations[rel] = max(collapse_relations[rel], score)
            else:
                collapse_relations[rel] = score

        documents[doc["doc_id"]]["predicted_n_ary_relations"] = collapse_relations

    precision = defaultdict(list)
    recall = defaultdict(list)

    for i, d in enumerate(documents):
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
                m = r
                if m in invert_subrelations:
                    new_relation.append(invert_subrelations[m])
                else:
                    new_relation.append([m])

            new_relations = list(product(*new_relation))
            for r in new_relations:
                mapped_relations[tuple(r)] = max(mapped_relations[tuple(r)], score)

        predicted_relations = set([tuple(k) for k, v in mapped_relations.items() if v == 1])
        gold_relations = set([tuple([r[e] for e in used_entities]) for r in gold_relations])

        for elist in combinations(used_entities, args.card):
            prel = [tuple([r for i, r in enumerate(p) if used_entities[i] in elist]) for p in predicted_relations]
            prel = set([p for p in prel if all(x is not None for x in p)])
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

    breakpoint()
    print(pd.DataFrame(metrics).T.to_latex(float_format=lambda x: "{:0.3f}".format(x)))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
