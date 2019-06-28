import json
import pandas as pd
from tqdm import tqdm
from allennlp.data.dataset_readers.dataset_utils.span_utils import bioul_tags_to_spans

from typing import List, Dict

tqdm.pandas()

available_entity_types_sciERC = ['Material', 'Metric', 'Task', 'Generic', 'OtherScientificTerm', 'Method']
map_available_entity_to_true = {
    'Material' : 'dataset',
    'Metric' : 'metric',
    'Task' : 'task',
    'Method' : 'model_name'
}

map_true_entity_to_available = {v:k for k, v in map_available_entity_to_true.items()}
used_entities = map_available_entity_to_true.keys()

def get_spans(taglist, wordlist) :
    entities = {k:[] for k in available_entity_types_sciERC}
    spans = bioul_tags_to_spans(taglist)
    for enttype, (start, end) in spans :
        entities[enttype].append([start, end+1, " ".join(wordlist[start:end+1])])
            
    return entities

def extract_entities_from_sentence(row) :
    entity_spans = get_spans(row['tags'], row['words'])
    return pd.Series(entity_spans)

def extract_entites_from_document(row) :
    for enttype in available_entity_types_sciERC:
        row[enttype] = [x for y in row[enttype] for x in y]
    return row

def load_pwc_sentence_predictions() :
    pwc_sentences = [json.loads(line) for line in open('data/pwc_s2_cleaned_text_v2_sentences.jsonl')]

    pwc_output = []
    for line in tqdm(open('outputs/pwc_s2_cleaned_text_v2_sentences_predictions.jsonl')) :
        line = json.loads(line)
        del line['logits']
        del line['mask']
        pwc_output.append(line)

    for s, t in zip(pwc_sentences, pwc_output) :
        s.update(t)

    pwc_sentences = pd.DataFrame(pwc_sentences)
    del pwc_output
    
    pwc_sentences[available_entity_types_sciERC] = pwc_sentences.progress_apply(extract_entities_from_sentence, axis=1)
    
    return pwc_sentences

def load_pwc_full_text() :
    pwc_df = pd.read_json('data/pwc_s2_cleaned_text_v2.jsonl', lines=True)
    return pwc_df

def get_pwc_data_and_output() :
    pwc_df = load_pwc_full_text()
    pwc_sentences = load_pwc_sentence_predictions()
    pwc_sentences = pwc_sentences.groupby(['doc_id'], as_index=False) \
                                 .aggregate(lambda x : tuple(x)) \
                                 .apply(extract_entites_from_document, axis=1)
    
    ids_to_keep = open('data/train_doc_ids.txt').read().split('\n') + open('data/dev_doc_ids.txt').read().split('\n')
    pwc_df = pwc_df[pwc_df.s2_paper_id.isin(ids_to_keep)]
    pwc_sentences = pwc_sentences[pwc_sentences.doc_id.isin(ids_to_keep)]

    return pwc_df, pwc_sentences[['doc_id', 'sentence'] + available_entity_types_sciERC]

##########################################################################################################################
###### Analyse preliminary results 

def get_aggregated_field(pwc_df, field) :
    df_agg = pwc_df.groupby(['s2_paper_id'])[field].aggregate(lambda x : tuple(set(list(x)))).reset_index()
    return df_agg

def compare_true_and_predicted_field(pwc_df, pwc_sentences, true_field, predicted_field, comparison_method) :
    predicted_df = pwc_sentences[['doc_id', predicted_field, 'sentence']]
    true_df = get_aggregated_field(pwc_df, true_field)
    joined = true_df.merge(predicted_df, left_on=['s2_paper_id'], right_on=['doc_id'])
    
    lowercase = lambda x : [str(w[2]).lower() for w in x]
    
    joined['sentence'] = joined['sentence'].apply(lambda x : " ".join(x).lower())
    joined[predicted_field] = joined[predicted_field].apply(lambda x : lowercase(x) if x == x else [])
    joined[true_field] = joined[true_field].apply(lambda x : [str(w).lower() for w in x])
     
    def compare_fields(row) :
        row['in_full'] = sum(tuple([x in row['sentence'] for x in row[true_field]]))
        row['in_predicted'] = sum(tuple([comparison_method(x, row[predicted_field]) for x in row[true_field]]))
        row['total_predicted'] = len(row[predicted_field])
        row['total_true'] = len(row[true_field])
        return row
    
    joined = joined.apply(compare_fields, axis=1)[['doc_id', 'in_full', 'in_predicted', 'total_true', 'total_predicted']]\
                   .set_index('doc_id', drop=True)

    return joined

def get_result_matrix_for_match_method(pwc_df, pwc_sentences, comparison_method) :
    predicted_entities = ['Material', 'Metric', 'Task', 'Method', 'Generic', 'OtherScientificTerm']
    true_entities = ['metric', 'model_name', 'task', 'score', 'dataset']

    results = []

    for pe in tqdm(predicted_entities) :
        for te in tqdm(true_entities) :
            aggregated_dataset_values = compare_true_and_predicted_field(pwc_df, pwc_sentences, te, pe, comparison_method).sum()
            aggregated_dataset_values['predicted'] = pe
            aggregated_dataset_values['true'] = te
            results.append(aggregated_dataset_values)

    results = pd.DataFrame(results)

    results['recall_oracle'] = results['in_full'] / results['total_true'] * 100
    results['recall_predicted'] = results['in_predicted'] / results['total_true'] * 100
    results['precision_predicted'] = results['in_predicted'] / results['total_predicted'] * 100

    return results

