import sys
sys.path.insert(0, '')
from scripts.entity_utils import *
from itertools import combinations
import numpy as np
import json

is_x_in_y = lambda x, y: x[0] >= y[0] and x[1] <= y[1]


def process_rows_for_doc(rows):
    doc_id = rows.name
    try :
        rows = rows.sort_values(by=["para_num", "sentence_num"])
    except :
        breakpoint()
    n_paragraphs = rows["para_num"].max() + 1

    n_sections = rows['section_id'].max() + 1
    sections = [0 for _ in range(n_sections)]
    paragraphs = [0 for _ in range(n_paragraphs)]

    words = []
    ner = []
    relations = []
    coreference = {}

    rows = rows.to_dict("records")
    coreference = {k: [] for e in used_entities for k in rows[0][e + "_Rel"]}

    section_heads = [0] * n_sections

    for index, row in enumerate(rows):
        entities = [
            [len(words) + e.token_start, len(words) + e.token_end, e.entity + "_" + (str(len(e.links) > 0)), e]
            for e in row["entities"]
        ]

        words += row["words"]
        ner += entities
        paragraphs[row["para_num"]] += len(row["words"])
        sections[row['section_id']] += len(row['words'])

        if row['para_id'] == 0 and row['sentence_id'] == 0 :
            section_heads[row['section_id']] = row['words']


        for e in entities:
            for k in e[-1].links:
                coreference[k].append(e[:2])

    ner = sorted(ner, key=lambda x: (x[0], x[1]))

    for e1, e2 in combinations(ner, 2):
        if e1[-1].entity != e2[-1].entity and len(e1[-1].links) > 0 and len(e2[-1].links) > 0:
            t1 = set().union(*[rows[0][e1[-1].entity + "_Rel"][k] for k in e1[-1].links])
            t2 = set().union(*[rows[0][e2[-1].entity + "_Rel"][k] for k in e2[-1].links])
            if len(t1 & t2) > 0:
                relations.append([e1[:2], e2[:2]])

    for i, e in enumerate(ner):
        ner[i] = e[:-1]

    para_ends = np.cumsum(paragraphs)
    para_starts = para_ends - np.array(paragraphs)

    paragraphs = list(zip(list(para_starts), list(para_ends)))

    section_ends = np.cumsum(sections)
    section_starts = section_ends - np.array(sections)

    sections = list(zip(list(section_starts), list(section_ends)))

    for e in ner:
        assert any([is_x_in_y(e, x) for x in paragraphs])

    n_ary_relations = rows[0]["Relations"]

    return {
        "paragraphs": paragraphs,
        "sections" : sections,
        "section_heads" : section_heads,
        "words": words,
        "ner": ner,
        "coref": coreference,
        "relations": relations,
        "doc_id": doc_id,
        "n_ary_relations": n_ary_relations,
    }


def read_dataframe(df_concat):
    """
    df_concat.columns = Index(['doc_id', 'para_id', 'section_id', 'sentence_id', 'sentence', 'words',
    'entities', 'stats', 'Relations', 'Material_Rel', 'Method_Rel',
    'Metric_Rel', 'Task_Rel', 'para_num', 'sentence_num'],
    dtype='object')
    """

    return df_concat.groupby("doc_id").progress_apply(process_rows_for_doc)


from sklearn.model_selection import train_test_split
import os


def dump_to_file(data, output_dir, max_id, test_size=0.3, random_state=1001):
    data = data[data.index < max_id]
    data.sort_index(inplace=True)
    split_data = {}
    split_data["train"], remain_data = train_test_split(data, test_size=test_size, random_state=random_state)
    split_data["dev"], split_data["test"] = train_test_split(remain_data, test_size=0.5, random_state=random_state)

    os.makedirs(output_dir, exist_ok=True)
    for split in split_data:
        split_data[split].to_json(os.path.join(output_dir, split + ".jsonl"), orient="records", lines=True)


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
    pwc_instance["coref"] = {
        k:v for k, v in pwc_instance['coref'].items() if len(v) > 0
    }
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

if __name__ == '__main__' :
    dump_sciERC_to_file('../data/sciERC_processed_data/json', '../data/sciERC_processed_data/json_pwc_format/')