from scripts.analyse_pwc_entity_results import *
from itertools import combinations
import numpy as np

is_x_in_y = lambda x, y : x[0] >= y[0] and x[1] <= y[1]

def process_rows_for_doc(rows):
    doc_id = rows.name
    rows = rows.sort_values(by='para_num')
    n_paragraphs = rows["para_num"].max() + 1
    paragraphs = [0 for _ in range(n_paragraphs)]
    words = []
    ner = []
    relations = []
    coreference = {}

    rows = rows.to_dict("records")
    coreference = {k:[] for e in used_entities for k in rows[0][e + '_Rel']}

    for index, row in enumerate(rows):
        entities = [
            [len(words) + e.token_start, len(words) + e.token_end, e.entity + "_" + (str(len(e.links) > 0)), e]
            for e in row['entities']
        ]

        words += row["words"]
        ner += entities
        paragraphs[row['para_num']] += len(row['words'])

        for e in entities :
            for k in e[-1].links :
                coreference[k].append(e[:2])

    ner = sorted(ner, key=lambda x : (x[0], x[1]))
    
    for e1,e2 in combinations(ner, 2) :
        if e1[-1].entity != e2[-1].entity and len(e1[-1].links) > 0 and len(e2[-1].links) > 0:
            t1 = set().union(*[rows[0][e1[-1].entity + '_Rel'][k] for k in e1[-1].links])
            t2 = set().union(*[rows[0][e2[-1].entity + '_Rel'][k] for k in e2[-1].links])
            if len(t1 & t2) > 0 :
                relations.append([e1[:2], e2[:2]])

    for i, e in enumerate(ner) :
        ner[i] = e[:-1]

    para_ends = np.cumsum(paragraphs)
    para_starts = para_ends - np.array(paragraphs)

    paragraphs = list(zip(list(para_starts), list(para_ends)))

    for e in ner :
        assert any([is_x_in_y(e, x) for x in paragraphs])

    n_ary_relations = rows[0]['Relations']
        
    return {'paragraphs' : paragraphs, 
            'words' : words, 
            'ner' : ner, 
            'coref' : coreference, 
            'relations' : relations, 
            'doc_id' : doc_id,
            'n_ary_relations' : n_ary_relations}

def read_dataframe(df_concat):
    """
    df_concat.columns = Index(['doc_id', 'para_id', 'section_id', 'sentence_id', 'sentence', 'words',
    'entities', 'stats', 'Relations', 'Material_Rel', 'Method_Rel',
    'Metric_Rel', 'Task_Rel', 'para_num', 'sentence_num'],
    dtype='object')
    """

    return df_concat.groupby('doc_id').progress_apply(process_rows_for_doc)

from sklearn.model_selection import train_test_split
import os
def dump_to_file(data, output_dir, max_id, test_size=0.3, random_state=1001) :
    data = data[data.index < max_id]
    data.sort_index(inplace=True)
    split_data = {}
    split_data['train'], remain_data = train_test_split(data, test_size=test_size, random_state=random_state)
    split_data['dev'], split_data['test'] = train_test_split(remain_data, test_size=0.5, random_state=random_state)

    os.makedirs(output_dir, exist_ok=True)
    for split in split_data :
        split_data[split].to_json(os.path.join(output_dir, split + '.jsonl'), orient='records', lines=True)
    