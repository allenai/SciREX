import argparse
import json
import os
import sys

sys.path.insert(0, "")


from scripts.entity_utils import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm

is_x_in_y = lambda x, y: x[0] >= y[0] and x[1] <= y[1]


parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, required=True)
parser.add_argument('--input_file', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
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
    pwc_instance["ner"] = [
        [e[0], e[1] + 1, e[2] + "_False"] for s in instance["ner"] for e in s if e[-1] in used_entities
    ]
    entities_used = [(e[0], e[1]) for s in instance["ner"] for e in s if e[-1] in used_entities]
    pwc_instance["coref"] = {
        "Cluster_" + str(i): [[e[0], e[1] + 1] for e in c if tuple(e) in entities_used]
        for i, c in enumerate(instance["clusters"])
    }
    pwc_instance["coref"] = {k: v for k, v in pwc_instance["coref"].items() if len(v) > 0}
    pwc_instance["relations"] = []
    pwc_instance["n_ary_relations"] = []
    pwc_instance["doc_id"] = instance["doc_key"]

    return pwc_instance


def dump_sciERC_to_file(scierc_input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for split in ["train", "dev", "test"]:
        data = [json.loads(line) for line in open(os.path.join(scierc_input_dir, split + ".json"))]
        data = [make_sciERC_into_pwc_format(ins) for ins in data]
        f = open(os.path.join(output_dir, split + ".jsonl"), "w")
        f.write("\n".join([json.dumps(ins) for ins in data]))
        f.close()

from dygie.data.dataset_readers.pwc_json import clean_json_dict

def convert_to_scierc(instance) :
    instance = clean_json_dict(instance)
    entities = instance['ner']
    corefs = instance['coref']
    n_ary_relations = instance['n_ary_relations']

    sections = instance['sections']
    sentences = instance['sentences']
    words = instance['words']
    words_grouped = [[words[s:e] for s, e in sent_list] for sent_list in sentences]
    entities_grouped = [[[] for _ in range(len(sent_list))] for sent_list in sentences]
    for e in entities :
        index = [
            (i, j) for i, sents in enumerate(sentences) for j, sspan in enumerate(sents) if is_x_in_y(e, sspan)
        ]
        assert len(index) == 1, breakpoint()
        i, j = index[0]
        entities_grouped[i][j].append((e[0], e[1], entities[e][1]))

    coreference = []
    relations = []

    span_to_cluster_ids = {}
    for cluster_name in corefs:
        for span in corefs[cluster_name]:
            span_to_cluster_ids.setdefault(tuple(span), []).append(cluster_name)

    span_to_cluster_ids = {span: set(sorted(v)) for span, v in span_to_cluster_ids.items()}

    cluster_to_relation = {}

    for rel_idx, rel in enumerate(n_ary_relations):
        for entity in used_entities:
            cluster_to_relation.setdefault(rel[entity], []).append(rel_idx)
    
    cluster_to_relation = {c:set(v) for c, v in cluster_to_relation.items()}

    for sec_entities in entities_grouped :
        relation_sec = []
        coreference_sec = {k:[] for k in corefs.keys()}
        for entities in sec_entities :
            relation_sentence = []
            for i in range(len(entities)) :
                for j in range(len(entities)) :
                    if i < j or entities[i][2] == entities[j][2] :
                        continue
                    
                    span_1, span_2 = (entities[i][0], entities[i][1]), (entities[j][0], entities[j][1])
                    c1, c2 = span_to_cluster_ids.get(span_1, set()), span_to_cluster_ids.get(span_2, set())
                    if len(set.union(set(), *[cluster_to_relation[c] for c in c1]) & set.union(set(), *[cluster_to_relation[c] for c in c2])) > 0:
                        relation_sentence.append([span_1[0], span_1[1], span_2[0], span_2[1], 'USED-FOR'])

                    if len(c1) > 0 :
                        c1 = list(sorted(c1))[0]
                        coreference_sec[c1].append(span_1)

                    if len(c2) > 0 :
                        c2 = list(sorted(c2))[0]
                        coreference_sec[c2].append(span_2)

            relation_sec.append(relation_sentence)

        coreference.append([list(set(cluster)) for cluster in list(coreference_sec.values())])
        relations.append(relation_sec)

    scierc_data = []
    for i in range(len(sections)) :
        data = {}
        data['sentences'] = words_grouped[i]
        data['ner'] = entities_grouped[i]
        data['clusters'] = coreference[i]
        data['relations'] = relations[i]
        data['doc_key'] = instance['doc_id'] + '_sec:' + str(i)
        data['doc_id'] = instance['doc_id']
        scierc_data.append(data)

    return scierc_data

from scripts.convert_brat_annotations_to_json import load_jsonl, annotations_to_jsonl
def convert_to_scierc_all(file, output_file) :
    data = load_jsonl(file)
    scierc_data = []
    for d in tqdm(data) :
        scierc_data += convert_to_scierc(d)

    annotations_to_jsonl(scierc_data, output_file)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.type == 'scierc' :
        convert_to_scierc_all(args.input_file, args.output_dir)

    # dump_to_file(args.input_file, args.output_dir, args.max_id)
