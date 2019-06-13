import json
import pandas as pd
from tqdm import tqdm
from allennlp.data.dataset_readers.dataset_utils.span_utils import bioul_tags_to_spans

from typing import List, Dict

def get_spans(taglist, wordlist) :
    entities = {}
    spans = bioul_tags_to_spans(taglist)
    for enttype, (start, end) in spans :
        if enttype not in entities :
            entities[enttype] = []
        entities[enttype].append(" ".join(wordlist[start:end+1]))
            
    return entities

def extract_entites(row) :
    document_entities = {}
    sentences = row['words']
    tags = row['tags']
    for wordlist, taglist in zip(sentences, tags) :
        entities = get_spans(taglist, wordlist)
        for enttype, entlist in entities.items() :
            if enttype not in document_entities :
                document_entities[enttype] = []
            document_entities[enttype] += entlist
    return pd.Series(document_entities)

def get_pwc_data_and_output() :
    pwc_df_with_dataset = [json.loads(line) for line in open('/net/nfs.corp/s2-research/beltagy/result_extraction/data/pwc_s2.jsonl')]
    pwc_df = [json.loads(line) for line in open('data/pwc_s2.jsonl')]

    pwc_df = pd.DataFrame(pwc_df)
    pwc_df_with_dataset = pd.DataFrame(pwc_df_with_dataset)
    pwc_df['dataset'] = pwc_df_with_dataset['dataset']

    pwc_sentences = [json.loads(line) for line in open('data/pwc_sentences.jsonl')]

    pwc_output = []
    for line in tqdm(open('outputs/pwc_sentences_predict_total.jsonl')) :
        line = json.loads(line)
        del line['logits']
        del line['mask']
        pwc_output.append(line)

    for s, t in zip(pwc_sentences, pwc_output) :
        s.update(t)

    pwc_sentences = pd.DataFrame(pwc_sentences)
    del pwc_output

    pwc_sentences = pwc_sentences.groupby(['doc_id', 'text_field'], as_index=False).aggregate(lambda x : tuple(x))

    pwc_sentences_entities = pwc_sentences.apply(extract_entites, axis=1)
    pwc_sentences = pd.concat([pwc_sentences, pwc_sentences_entities], axis=1)

    return pwc_df, pwc_sentences

def get_aggregated_field(pwc_df, field) :
    df_agg = pwc_df.groupby(['s2_paper_id'])[field].aggregate(lambda x : tuple(set(list(x)))).reset_index()
    return df_agg

def exact_match_in_concatenated(true_value: str, predicted_value_list: List[str]) :
    return true_value in " ".join(predicted_value_list)

def exact_match_with_any(true_value: str, predicted_value_list: List[str]) :
    return true_value in predicted_value_list

def exact_match_in_any(true_value: str, predicted_value_list: List[str]) :
    return any([true_value in x for x in predicted_value_list])

def compare_true_and_predicted_field(pwc_df, pwc_sentences, true_field, predicted_field, comparison_method) :
    predicted_df = pwc_sentences[['doc_id', 'text_field', predicted_field, 'sentence']]
    true_df = get_aggregated_field(pwc_df, true_field)
    joined = true_df.merge(predicted_df, left_on=['s2_paper_id'], right_on=['doc_id'])
    
    lowercase = lambda x : [str(w).lower() for w in x]
    
    joined['sentence'] = joined['sentence'].apply(lambda x : " ".join(x).lower())
    joined[predicted_field] = joined[predicted_field].apply(lambda x : lowercase(x) if x == x else [])
    joined[true_field] = joined[true_field].apply(lowercase)
     
    def compare_fields(row) :
        row['in_full'] = tuple([x in row['sentence'] for x in row[true_field]])
        row['in_predicted'] = tuple([comparison_method(x, row[predicted_field]) for x in row[true_field]])
        row['n_predicted'] = len(row[predicted_field])
        return row
    
    joined = joined.apply(compare_fields, axis=1)

    return joined

def aggregate_across_text_field(oracle) :
    def collapse(rows) :
        in_full = [any(x) for x in zip(*list(rows['in_full']))]
        in_pred = [any(x) for x in zip(*list(rows['in_predicted']))]
        total_predicted = sum(rows['n_predicted'])
        
        return pd.Series({'in_full' : sum(in_full), 'in_predicted' : sum(in_pred), 'total_true' : len(in_full), 'total_predicted' : total_predicted})
    oracle = oracle.groupby(['s2_paper_id', 'doc_id'])[['in_full', 'in_predicted', 'n_predicted']].apply(collapse)
    return oracle

def get_result_matrix_for_match_method(pwc_df, pwc_sentences, comparison_method) :
    predicted_entities = ['Material', 'Metric', 'Task', 'Method', 'Generic', 'OtherScientificTerm']
    true_entities = ['metric', 'model_name', 'task', 'score', 'dataset']

    results = []

    for pe in tqdm(predicted_entities) :
        for te in tqdm(true_entities) :
            oracle = compare_true_and_predicted_field(pwc_df, pwc_sentences, te, pe, comparison_method)
            aggregated_dataset_values = aggregate_across_text_field(oracle).sum()
            aggregated_dataset_values['predicted'] = pe
            aggregated_dataset_values['true'] = te
            results.append(aggregated_dataset_values)

    results = pd.DataFrame(results)

    results['recall_oracle'] = results['in_full'] / results['total_true'] * 100
    results['recall_predicted'] = results['in_predicted'] / results['total_true'] * 100
    results['precision_predicted'] = results['in_predicted'] / results['total_predicted'] * 100

    return results

