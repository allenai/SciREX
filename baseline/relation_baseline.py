from typing import List, Tuple, Dict
from scripts.analyse_pwc_entity_results import *
import pandas as pd
import numpy as np

from dygie.training.pwc_relation_metrics import RelationMetrics

def generate_relations(words: List[str], spans: List[Tuple[int, int, str]], coref: Dict[str, List[Tuple[int, int]]]):
    spans = sorted(spans, key=lambda x: (x[0], x[1]))

    last_linked_spans = {k: None for k in used_entities}

    predicted_relations = {}
    predicted_links = {}

    inverted_coref = {tuple(s): [] for k, v in coref.items() for s in v}
    for k, v in coref.items():
        for s in v:
            inverted_coref[tuple(s)].append(k)

    for span in spans:
        start, end, label = span
        entity_type, is_linked = label.split("_")
        is_linked = is_linked == str(True)

        if is_linked:
            for pre_entity_type in used_entities:
                if pre_entity_type != entity_type and last_linked_spans[pre_entity_type] is not None:
                    pre_start, pre_end, pre_label = last_linked_spans[pre_entity_type]
                    pre_span_coreferents = list(
                        set([tuple(x) for cluster in inverted_coref[(pre_start, pre_end)] for x in coref[cluster]])
                    )
                    span_coreferents = list(
                        set([tuple(x) for cluster in inverted_coref[(start, end)] for x in coref[cluster]])
                    )
                    for p, q in product(pre_span_coreferents, span_coreferents):
                        if p < q:
                            predicted_relations[(p, q)] = pre_entity_type + "-" + entity_type
                        else:
                            predicted_relations[(q, p)] = entity_type + "-" + pre_entity_type

                    pre_links = inverted_coref[(pre_start, pre_end)]
                    post_links = inverted_coref[(start, end)]

                    for p, q in product(pre_links, post_links):
                        if (pre_entity_type, entity_type) in binary_relations:
                            predicted_links[(p, q)] = pre_entity_type + "-" + entity_type
                        else:
                            predicted_links[(q, p)] = entity_type + "-" + pre_entity_type

            last_linked_spans[entity_type] = span

    return predicted_relations, predicted_links

def f1(p, r) :
    return (2*p*r)/(p + r)

def compute_metrics(instances):
    n_true, n_pred, n_match = 0, 0, 0
    link_metrics = RelationMetrics([r[0]+'-'+r[1] for r in binary_relations])

    link_metrics_oracle = RelationMetrics([r[0]+'-'+r[1] for r in binary_relations])
    link_metrics_all = RelationMetrics([r[0]+'-'+r[1] for r in binary_relations])

    document_scores = []
    for instance in tqdm(instances):
        predicted_relations, predicted_links = generate_relations(instance["words"], instance["ner"], instance["coref"])

        true_relations = instance["relations"]
        entities = {(x[0], x[1]): x[2].split("_")[0] for x in instance["ner"]}
        gold_relations = {}
        for p, q in true_relations:
            p, q = sorted([tuple(p), tuple(q)])
            gold_relations[(p, q)] = entities[p] + "-" + entities[q]

        gold_links = {}
        map_link_to_type = {}
        for rel in instance["n_ary_relations"]:
            rel = Relation(*rel)._asdict()
            for r1 in used_entities :
                map_link_to_type[rel[r1]] = r1
            for r1, r2 in binary_relations:
                gold_links[(rel[r1], rel[r2])] = r1 + "-" + r2

        oracle_links = {}
        all_links = {}
        present_entities = [k for k, v in instance['coref'].items() if len(v) > 0]
        for k in present_entities :
            for k1 in present_entities :
                if (k, k1) in gold_links :
                    oracle_links[(k, k1)] = gold_links[(k, k1)]

                t = map_link_to_type[k]
                t1 = map_link_to_type[k1]
                if t != t1 :
                    if (t, t1) in binary_relations :
                        all_links[(k, k1)] = t + '-' + t1
                    else :
                        all_links[(k1, k)] = t1 + '-' + t


        link_metrics([predicted_links], [gold_links])
        link_metrics_oracle([oracle_links], [gold_links])
        link_metrics_all([all_links], [gold_links])

        n_true += len(gold_relations)
        n_pred += len(predicted_relations)

        for key in predicted_relations:
            if key in gold_relations:
                n_match += 1

        doc_n_match = 0
        for key in all_links :
            if key in gold_links :
                doc_n_match += 1

        doc_n_match_head = 0
        for key in predicted_links :
            if key in gold_links :
                doc_n_match_head += 1

        doc_n_true = len(gold_links)
        doc_n_pred = len(all_links)
        document_scores.append({
            'true' : doc_n_true,
            'pred_all' : doc_n_pred,
            'match_all' : doc_n_match,
            'pred_head' : len(predicted_links),
            'match_head' : doc_n_match_head,
            'n_ent' : len(map_link_to_type)
        })

    precision = n_match / n_pred
    recall = n_match / n_true

    precisiong_class, recallg_class, f1g_class = link_metrics.get_metric(reset=True)
    precisionoracle_class, recalloracle_class, f1oracle_class = link_metrics_oracle.get_metric(reset=True)
    precisionall_class, recallall_class, f1all_class = link_metrics_all.get_metric(reset=True)

    return (
        (precision, recall, (2 * precision * recall) / (precision + recall)),
        pd.DataFrame({'precision' : precisiong_class, 'recall' : recallg_class, 'f1' : f1g_class}),
        pd.DataFrame({'precision' : precisionoracle_class, 'recall' : recalloracle_class, 'f1' : f1oracle_class}),
        pd.DataFrame({'precision' : precisionall_class, 'recall' : recallall_class, 'f1' : f1all_class}),
        document_scores
    )

