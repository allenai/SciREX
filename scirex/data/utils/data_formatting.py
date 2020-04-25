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

