import os

import numpy as np
import pandas as pd
from tqdm import tqdm

def process_folder(folder) :
    span_labels = {}
    map_T_to_span = {}
    if not os.path.isdir(folder) or 'document.txt' not in os.listdir(folder) :
        print(folder, " have not document")
        return None
    doc_file = open(os.path.join(folder, 'document.txt')).read()
    ann_file = open(os.path.join(folder, 'document.ann')).read().strip()
    annotations = ann_file.split('\n')
    for ann in annotations :
        ann_type, ann = ann.split('\t', 1)
        if ann_type[0] == 'T' :
            ann, ann_text = ann.split('\t')
            enttype, span_start, span_end = ann.split()
            span_start, span_end = int(span_start), int(span_end)
            if (span_start, span_end) in span_labels :
                assert "Span already present"
            else :
                span_labels[(span_start, span_end)] = {'E' : enttype, 'A' : [], 'T' : ann_text}
                map_T_to_span[ann_type] = (span_start, span_end)
        if ann_type[0] == 'A' :
            ann, ann_T = ann.split()
            if ann_T in map_T_to_span :
                span_labels[map_T_to_span[ann_T]]['A'].append(ann)
            else :
                assert "Attribute before Trigger"
    return span_labels, doc_file

def get_all_document_annotations(brat_folder) :
    map_id_to_ann = {}
    for f in tqdm(os.listdir(brat_folder)) :
        map_id_to_ann[f] = process_folder(os.path.join(brat_folder, f))
    return map_id_to_ann

def process_back_to_dataframe(span_labels, doc_text) :
    sentences = doc_text.split('\n ')
    assert sentences[-1] == ''
    sentences = [x + '\n ' for x in sentences[:-1]]
    sentence_limits = np.cumsum([len(x) for x in sentences])
    sentence_limits = list(zip([0] + list(sentence_limits)[:-1], sentence_limits))
    for s, e in sentence_limits :
        assert doc_text[e-2:e] == '\n '
        assert doc_text[s] != ' '
        
    span_labels = list(map(lambda x: [list(x[0]), x[1]], sorted(span_labels.items(), key=lambda x: x[0][0])))
    sl_ix = 0
    map_sentence_limits_to_spans = {}
    for ss, se in sentence_limits :
        map_sentence_limits_to_spans[(ss, se)] = []
        while sl_ix < len(span_labels) and span_labels[sl_ix][0][0] >= ss and span_labels[sl_ix][0][1] <= se:
            map_sentence_limits_to_spans[(ss, se)].append(span_labels[sl_ix])
            sl_ix += 1
           
    spans_in_l = 0
    for k, v in map_sentence_limits_to_spans.items() :
        for span, _ in v :
            assert k[0] <= span[0] and k[1] >= span[1]
            spans_in_l += 1
            assert span[1] < k[1] - 1
    assert spans_in_l == len(span_labels)
    
    for k, v in map_sentence_limits_to_spans.items() :
        for span, _ in v :
            span[0] -= k[0]
            span[1] -= k[0]

    df = []
    for sent_id, ((ss, se), st) in enumerate(zip(sentence_limits, sentences)) :
        for span, d in map_sentence_limits_to_spans[(ss, se)] :
            assert st[-2:] == '\n ', st[-2:]
            assert span[1] < len(st) - 2
            assert st[span[0]:span[1]] == d['T'] and len(d['T']) > 0, (st[span[0]:span[1]], d['T'])
        df.append({'sentence' : st, 'spans' : map_sentence_limits_to_spans[(ss, se)], 'sentence_id' : sent_id})
    
    return pd.DataFrame(df)

def get_dataframe_from_folder(brat_folder) :
    map_changes = get_all_document_annotations(brat_folder)

    doc_df = []
    for k in map_changes :
        if map_changes[k] is None :continue
        df = process_back_to_dataframe(*map_changes[k])
        df['doc_id'] = k
        doc_df.append(df)

    doc_df = pd.concat(doc_df)
    return doc_df
