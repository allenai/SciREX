from scirex_utilities.json_utilities import load_jsonl
from sys import argv
from collections import defaultdict
from scirex.predictors.utils import merge_method_subrelations, span_match
from scirex_utilities.entity_utils import used_entities
from itertools import combinations

from scirex.evaluation_scripts.scirex_relation_evaluate import has_all_mentions
import pandas as pd


def convert_instance(instance, start) :
    instance['predicted_ner'] = [(s + start, e + start + 1, t) for sent in instance['predicted_ner'] for (s, e, t) in sent]
    instance['predicted_relations'] = [((s1 + start, e1 + start + 1), (s2 + start, e2 + start + 1)) for sent in instance['predicted_relations'] for (s1, e1, s2, e2, t) in sent]
    return instance

def combine_same_doc_instances(instances) :
    starting_idx = 0
    document = {'ner' : [], 'relations' : []}
    for i, instance in enumerate(instances) :
        assert i == instance['section_id']
        instances[i] = convert_instance(instance, starting_idx)
        starting_idx += sum([len(s) for s in instance['sentences']])
        document['ner'] += instances[i]['predicted_ner']
        document['relations'] += instances[i]['predicted_relations']
    
    return document

def convert_all_instances(instances) :
    documents = defaultdict(list)
    for instance in instances :
        doc_id = instance['doc_key'].split(':')[0]
        instance['section_id'] = int(instance['doc_key'].split(':')[1])
        documents[doc_id].append(instance)

    for doc in documents :
        documents[doc] = sorted(documents[doc], key=lambda x : x['section_id'])
        documents[doc] = combine_same_doc_instances(documents[doc])
        documents[doc]['doc_id'] = doc

    return documents

def evaluate(predicted_data, gold_data) :
    p, r, f1 = 0, 0, 0
    gold_data = {x['doc_id'] : x for x in gold_data}
    all_metrics = []

    for doc in predicted_data:
        predicted_doc = predicted_data[doc]
        gold_doc = gold_data[doc]
        merge_method_subrelations(gold_doc)
        gold_doc["clusters"] = gold_doc["coref"]

        gold_spans = [tuple(x) for x in gold_doc['ner']]
        predicted_spans = [tuple(x) for x in predicted_doc['ner']]

        for t in used_entities :
            typed_gold_spans = set([x for x in gold_spans if x[2] == t])
            typed_predicted_spans = set([x for x in predicted_spans if x[2] ==  t])

            matched = len(typed_gold_spans & typed_predicted_spans)
            tp, tr = matched / (len(typed_predicted_spans) + 1e-7), matched / (len(typed_gold_spans) + 1e-7)
            tf1 = 2*tp*tr / (tp + tr + 1e-7)

            p += tp / (len(used_entities) * len(predicted_data))
            r += tr / (len(used_entities) * len(predicted_data))
            f1 += tf1 / (len(used_entities) * len(predicted_data))

        clusters = gold_doc['coref']
        span_to_cluster = {}
        for c, spans in clusters.items() :
            for span in spans :
                span_to_cluster[tuple(span)] = c

        predicted_span_to_gold = {}
        for i, (s, e, t) in enumerate(predicted_spans) :
            span = (s, e)
            predicted_span_to_gold[span] = (t, span, str(i))
            for sg, eg, tg in gold_spans :
                span_g = (sg, eg)
                if span_match(span, span_g) > 0.5 :
                    predicted_span_to_gold[span] = (tg, span_g, span_to_cluster.get(span_g, str(i)))
                    break

        for types in combinations(used_entities, 2):
            gold_relations = [tuple((t, x[t]) for t in types) for x in gold_doc['n_ary_relations']]
            gold_relations = set([x for x in gold_relations if has_all_mentions(gold_doc, x)])

            predicted_relations = []
            for s1, s2 in predicted_doc['relations'] :
                if s1 in predicted_span_to_gold and s2 in predicted_span_to_gold :
                    t1, span_1, c_1 = predicted_span_to_gold[s1]
                    t2, span_2, c_2 = predicted_span_to_gold[s2]

                    if t1 in types and t2 in types and t1 != t2 :
                        rel = {t1: c_1, t2: c_2}
                        predicted_relations.append(tuple([(t, rel[t]) for t in types]))

            predicted_relations = set(predicted_relations)

            matched = predicted_relations & gold_relations
            metrics = {
                "p": len(matched) / (len(predicted_relations) + 1e-7),
                "r": len(matched) / (len(gold_relations) + 1e-7),
            }
            metrics["f1"] = 2 * metrics["p"] * metrics["r"] / (metrics["p"] + metrics["r"] + 1e-7)

            if len(gold_relations) > 0:
                all_metrics.append(metrics)


    print(p, r, f1)

    all_metrics = pd.DataFrame(all_metrics)
    print(f"Relation Metrics n={2}")
    print(all_metrics.describe().loc['mean'][['p', 'r', 'f1']])


if __name__ == '__main__' :
    evaluate(convert_all_instances(load_jsonl(argv[1])), load_jsonl(argv[2]))
 
    