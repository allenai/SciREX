import json
import numpy as np
import matplotlib.pyplot as plt
from scripts.entity_utils import *
import pandas as pd
import seaborn as sns

import os
import sys

from dygie.models.global_analysis.clustering import *
from dygie.models.global_analysis.relation_extraction import *

from dygie.training.evaluation import *
dev_data = [json.loads(line) for line in open('model_data/pwc_split_on_sectioned/dev.jsonl')]
cluster_matching_thresholds = generate_thresholds(dev_data)

def add_type_to_gold(x) :
    x['n_ary_relations'] = [Relation(*y)._asdict() for y in x['n_ary_relations']]
    x['true_entities'] = {}
    for entity in used_entities :
        x['true_entities'][entity] = list(set([rel[entity] for rel in x['n_ary_relations']]))
        
    x['gold_to_type'] = {}
    for rel in x['n_ary_relations'] :
        for k, v in rel.items() :
            x['gold_to_type'][v] = k
            
    x['gold_clusters'] = {k:{'spans' : v, 'words' : [" ".join(x['words'][y[0]:y[1]]) for y in v]} for k, v in x['coref'].items()}
            
    return x

def decoding(f1, f2) :
    dev_pred = [json.loads(line) for line in open(f1)]
    dev_data = [json.loads(line) for line in open(f2)]

    dev_data = {x['doc_id']:x for x in dev_data}
    for x in dev_pred :
        x['true_coref'] = dev_data[x['doc_id']]['coref']
        x['n_ary_relations'] = [Relation(*y)._asdict() for y in dev_data[x['doc_id']]['n_ary_relations']]
        x['true_entities'] = {}
        for entity in used_entities :
            x['true_entities'][entity] = list(set([rel[entity] for rel in x['n_ary_relations']]))

        x['gold_to_type'] = {}
        for rel in x['n_ary_relations'] :
            for k, v in rel.items() :
                x['gold_to_type'][v] = k
                
    hand_annotation = {}
                
    for v in dev_pred :
        v['missed'] = hand_annotation.get(v['doc_id'], {}).get('missed', [])
        v['table'] = hand_annotation.get(v['doc_id'], {}).get('table', [])
        
    for v in dev_pred :
        v['prediction'] = {tuple(x['span']):x['label'] for x in v['prediction']}
        v['gold'] = {tuple(x['span']):x['label'] for x in v['gold']}
        
    for ex in dev_pred :
        clusters, spl, cluster_labels, linked_clusters = do_clustering(ex, 'prediction', 'coref_prediction', plot=False)
        ex['clusters'] = clusters
        ex['linked_clusters'] = linked_clusters
        relation_matrix = get_relations_between_clusters(ex, 'prediction', cluster_labels) #, ex['relation_threshold'])
        ex['relation_matrix'] = relation_matrix
        ex['predicted_relations'] = {n:generate_relations(clusters, relation_matrix, linked_clusters, n, ex['relation_threshold'])
                                     for n in range(1, 6)}

        ex['gold_clusters'] = {k:{'spans' : v, 'words' : [" ".join(ex['words'][y[0]:y[1]]) for y in v]} for k, v in ex['true_coref'].items()}
        ex['true_relations'] = {n:generate_true_relations(ex['n_ary_relations'], n)
                                     for n in range(1, 6)}

        map_all_clusters_to_true(ex, cluster_matching_thresholds)
        
    oracle_res, oracle_exp_res, oracle_exp_res_table, gold_res, pred_res = run_eval_for_relation_extraction(dev_pred)
        
    return oracle_res, oracle_exp_res, oracle_exp_res_table, gold_res, pred_res

from tqdm import tqdm
if __name__ == '__main__' :
    results = prf_all(scores_scaffold(), n_d=1)
    print(results)
    for i in tqdm(range(0, 1080, 20)) :
        f1 = f'model_data/pwc_split_on_sectioned/{i}_unannotated.jsonl'
        f2 = f'outputs/unannotated_results_folder/{i}_unannotated/combined.jsonl'
        try :
            oracle_res, oracle_exp_res, oracle_exp_res_table, gold_res, pred_res = decoding(f2, f1)
            for n in pred_res :
                for t in pred_res[n] :
                    for k in pred_res[n][t] :
                        results[n][t][k] += pred_res[n][t][k]
        except :
            continue
                    
    for n, v in results.items() :
        if n < 5 :
            print(pd.DataFrame(pred_res[n]).mean(1)*100)
        