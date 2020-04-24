from scripts.convert_brat_annotations_to_json import load_jsonl
from scripts.entity_utils import used_entities

import sys
from tqdm import tqdm
from collections import defaultdict
from itertools import combinations, product

def clean_gold(json_dict) :
    corefs_all: Dict[str, List[Tuple[int, int]]] = json_dict["coref"]
    n_ary_relations_all: List[Dict[str, str]] = [x for x in json_dict["n_ary_relations"]]

    if "method_subrelations" in json_dict:
        method_subrelations: Dict[str, List[tuple]] = {
            x: [y[1] for y in v if y[1] != x] for x, v in json_dict["method_subrelations"].items()
        }

        for method_name, method_subparts in method_subrelations.items():
            for part in method_subparts:
                corefs_all[method_name].extend(corefs_all[part])

        for method_subparts in method_subrelations.values():
            for part in method_subparts:
                if part in corefs_all:
                    del corefs_all[part]

    for cluster, spans in corefs_all.items():
        corefs_all[cluster] = sorted(list(set([tuple(x) for x in spans])))

    # Remove clusters with no entries
    corefs = {k: v for k, v in corefs_all.items() if len(v) > 0}

    # Keep only relations where all clusters are non empty
    n_ary_relations = [r for r in n_ary_relations_all if all([v in corefs for k, v in r.items() if k in used_entities])]
    json_dict['coref'] = corefs
    json_dict['n_ary_relations'] = n_ary_relations
    return json_dict

def span_match(span_1, span_2) :
    sa, ea = span_1
    sb, eb = span_2
    iou = (min(ea, eb) - max(sa, sb)) / (max(eb, ea) - min(sa, sb))
    return iou

data = {x['doc_id']: clean_gold(x) for x in load_jsonl('../model_data/pwc_split_on_sectioned/dev.jsonl')}
predictions = load_jsonl('predictions/scierc_pwc_dev_all.jsonl')

def convert(instance) :
    doc_id, sec_id = instance['doc_key'].split(':')
    sec_id = int(sec_id)

    section_start = data[doc_id]['sections'][sec_id][0]

    predicted_ner = {(e[0]+section_start, e[1]+section_start):e[2] for sent in instance['predicted_ner'] for e in sent}
    words = [w for sent in instance['sentences'] for w in sent]
    relations = [[(r[0]+section_start, r[1]+section_start), (r[2]+section_start, r[3]+section_start), r[4]] for sent in instance['predicted_relations'] for r in sent]
    clusters = [[tuple(span) for span in cluster] for cluster in instance['predicted_clusters']]
    span_to_cluster = {}

    gold_clusters = {k:set([tuple(x) for x in v]) for k, v in data[doc_id]['coref'].items()}
    span_to_cluster_ids = defaultdict(list)
    for cluster_name in gold_clusters:
        for span in gold_clusters[cluster_name]:
            span_to_cluster_ids[tuple(span)].append(cluster_name)

    span_to_cluster_ids = {span: set(sorted(v)) for span, v in span_to_cluster_ids.items()}
    gold_ner = {(e[0], e[1]):e[2] for e in data[doc_id]['ner']}

    # if len(relations) > 0:
    #     print(len(relations))

    mapped_relations = []
    for r in relations :
        span_1, span_2, _ = r
        span_1 = [e for e in gold_ner if span_match(e, span_1) > 0.5]
        span_2 = [e for e in gold_ner if span_match(e, span_2) > 0.5]
        if len(span_1) == 0 or len(span_2) == 0 :
            continue

        span_1 = span_1[0]
        span_2 = span_2[0]
        span_1_clusters = list(span_to_cluster_ids.get(span_1, set()))
        span_2_clusters = list(span_to_cluster_ids.get(span_2, set()))
        for c1 in span_1_clusters :
            for c2 in span_2_clusters :
                relation = {k:None for k in used_entities}
                try :
                    relation[gold_ner[span_1]] = c1
                    relation[gold_ner[span_2]] = c2
                except :
                    breakpoint()
                mapped_relations.append(relation)

    # if len(mapped_relations) > 0:
    #     print(mapped_relations)
    
    return [doc_id, mapped_relations]
    
predicted_relations_list = {x:[] for x in data}
for p in tqdm(predictions) :
    doc_id, mapped_relations = convert(p)
    # print(doc_id, mapped_relations)
    predicted_relations_list[doc_id] += mapped_relations

# breakpoint()

precision = defaultdict(list)
recall = defaultdict(list)

import numpy as np
import pandas as pd

for i, d in enumerate(data):
    predicted_relations = predicted_relations_list[d]
    gold_relations = data[d]["n_ary_relations"]
    method_subrelations = data[d]["method_subrelations"]

    if len(gold_relations) == 0 :
        continue

    invert_subrelations = defaultdict(list)
    for m, ms in method_subrelations.items():
        for span, n in ms:
            invert_subrelations[n].append(m)

    mapped_relations = []
    for relation in predicted_relations:
        new_relation = {}
        for t, e in relation.items():
            if e in invert_subrelations:
                new_relation[t] = invert_subrelations[e]
            else:
                new_relation[t] = [e]

        for rel in product(*list(new_relation.values())) :
            mapped_relations.append(dict(zip(new_relation.keys(), rel)))

    try :
        predicted_relations = [tuple([r[e] for e in used_entities]) for r in mapped_relations]
        predicted_relations = set(predicted_relations)
    except :
        breakpoint()
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
