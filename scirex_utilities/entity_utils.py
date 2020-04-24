import os
from collections import namedtuple
from itertools import combinations

BASEPATH = os.getenv("RESULT_EXTRACTION_BASEPATH", ".")

available_entity_types_sciERC = ["Material", "Metric", "Task", "Generic", "OtherScientificTerm", "Method"]
map_available_entity_to_true = {"Material": "dataset", "Metric": "metric", "Task": "task", "Method": "model_name"}
map_true_entity_to_available = {v: k for k, v in map_available_entity_to_true.items()}

used_entities = list(map_available_entity_to_true.keys())
true_entities = list(map_available_entity_to_true.values())

Relation = namedtuple("Relation", used_entities + ["score"])

binary_relations = list(combinations(used_entities, 2))

def chunk_string(name) :
    c = ''
    start = 0
    chunks = []
    idx = []
    for i, w in enumerate(name) :
        if w not in ['+', '(', ')', ';'] :
            c += w
        else :
            chunks.append(c)
            idx.append([start, i])
            start = i + 1
            c = ''
            
    if c != '' :
        chunks.append(c)
        idx.append([start, len(name)])
        
    stripped_chunks = []
    stripped_idx = []
    for c, (s, e) in zip(chunks, idx) :
        nc = c.strip()
        ni = c.index(nc) + s
        if len(nc) == 0 :
            continue
        stripped_chunks.append(nc)
        stripped_idx.append([ni, ni + len(nc)])
        
    for c, (s, e) in zip(stripped_chunks, stripped_idx) :
        assert c == name[s:e], (c, name[s:e])
        
    return list(zip(stripped_idx, stripped_chunks))

import re
clean_name = lambda x : re.sub(r"[^\w-]", "_", x)

def _annotation_to_dict(dc):
    # convenience method
    if isinstance(dc, dict):
        ret = dict()
        for k, v in dc.items():
            k = _annotation_to_dict(k)
            v = _annotation_to_dict(v)
            ret[k] = v
        return ret
    elif isinstance(dc, str):
        return dc
    elif isinstance(dc, (set, frozenset, list, tuple)):
        ret = []
        for x in dc:
            ret.append(_annotation_to_dict(x))
        return tuple(ret)
    else:
        return dc

import json
def annotations_to_jsonl(annotations, output_file):
    with open(output_file, "w") as of:
        for ann in sorted(annotations, key=lambda x: x["doc_id"]):
            as_json = _annotation_to_dict(ann)
            as_str = json.dumps(as_json, sort_keys=True)
            of.write(as_str)
            of.write("\n")

def load_jsonl(filename) :
    with open(filename) as f :
        data = [json.loads(line) for line in f] 

    return data