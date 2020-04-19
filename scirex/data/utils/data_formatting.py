import argparse
import json
import os
import sys
from itertools import combinations

from scirex_utilities.entity_utils import used_entities, Relation
from scirex_utilities.convert_brat_annotations_to_json import load_jsonl
from sklearn.model_selection import train_test_split
from tqdm import tqdm

is_x_in_y = lambda x, y: x[0] >= y[0] and x[1] <= y[1]


parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, required=True)
parser.add_argument('--input_file', type=str, required=True)
parser.add_argument('--output_file', type=str, required=True)
parser.add_argument('--max_id', type=str, required=True)


def dump_to_file(input_file, output_dir, max_id, test_size=0.3, random_state=1001):
    data = load_jsonl(input_file)
    data = [x for x in data if x['doc_id'] < max_id]
    print("Leftover documents , ", len(data))
    data = sorted(data, key=lambda x : x['doc_id'])
    split_data = {}
    split_data["train"], remain_data = train_test_split(data, test_size=test_size, random_state=random_state)
    split_data["dev"], split_data["test"] = train_test_split(remain_data, test_size=0.5, random_state=random_state)

    os.makedirs(output_dir, exist_ok=True)
    for split in split_data:
        annotations_to_jsonl(split_data[split], os.path.join(output_dir, split + ".jsonl"))


def dump_all_to_file(input_file, output_dir):
    data = load_jsonl(input_file)
    print("Leftover documents , ", len(data))
    data = sorted(data, key=lambda x : x['doc_id'])

    os.makedirs(output_dir, exist_ok=True)
    for i in range(0, len(data), 20):
        annotations_to_jsonl(data[i:i+20], os.path.join(output_dir, str(i) + "_unannotated.jsonl"))
        json.dump(
            {"pwc": os.path.join(output_dir, str(i) + "_unannotated.jsonl")},
            open(os.path.join(os.path.dirname(output_dir), 'dataset_readers_paths/{i}_unannotated.json', "w")),
        )


########################


def make_sciERC_into_pwc_format(instance):
    pwc_instance = {}
    pwc_instance["words"] = [w for s in instance["sentences"] for w in s]
    pwc_instance["paragraphs"] = [[0, len(pwc_instance["words"])]]
    
    entities_used = {(e[0], e[1]): e[2] for s in instance["ner"] for e in s} # if e[-1] in used_entities]
    pwc_instance["coref"] = {
        "Cluster_" + str(i): [[e[0], e[1] + 1] for e in c if tuple(e) in entities_used]
        for i, c in enumerate(instance["clusters"])
    }

    for cluster, cluster_items in pwc_instance['coref'].items() :
        cluster_type = list(set([entities_used[tuple([e[0], e[1]-1])] for e in cluster_items]) & set(used_entities))
        if len(cluster_type) == 1 :
            cluster_type = cluster_type[0]
        else :
            cluster_type = [entities_used[tuple([e[0], e[1]-1])] for e in cluster_items][0]
        for e in cluster_items :
            entities_used[(e[0], e[1] - 1)] = cluster_type

    pwc_instance["ner"] = [
        [k[0], k[1] + 1, v + "_False"] for k, v in entities_used.items()
    ]
    pwc_instance["coref"] = {k: v for k, v in pwc_instance["coref"].items() if len(v) > 0}
    pwc_instance["relations"] = []
    pwc_instance["n_ary_relations"] = []
    pwc_instance["doc_id"] = instance["doc_id"]

    return pwc_instance


def dump_sciERC_to_file(scierc_input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for split in ["train", "dev", "test"]:
        data = [json.loads(line) for line in open(os.path.join(scierc_input_dir, split + ".json"))]
        data = [make_sciERC_into_pwc_format(ins) for ins in data]
        f = open(os.path.join(output_dir, split + ".jsonl"), "w")
        f.write("\n".join([json.dumps(ins) for ins in data]))
        f.close()

