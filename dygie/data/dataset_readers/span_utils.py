from allennlp.data.dataset_readers.dataset_utils.span_utils import to_bioul
from typing import List, Dict
from scripts.entity_utils import Relation

import numpy as np
from baseline.baseline import character_similarity_features

def spans_to_bio_tags(spans, length) :
    tag_sequence = ['O'] * length
    for span in spans :
        start, end, label = span
        tag_sequence[start] = 'B-' + label
        for ix in range(start + 1, end) :
            tag_sequence[ix] = 'I-' + label

    return to_bioul(tag_sequence, encoding='BIO')

def generate_seq_field(span_dict, length, element_map) :
    tag_sequence = [None]*length
    for (start, end), element in span_dict.items() :
        for j in range(start, end) :
            tag_sequence[j] = element

    tag_sequence = [element_map(x) for x in tag_sequence]
    return tag_sequence
        

def generate_only_span_text(js) :
    paragraphs: List[(int, int)] = js["sections"]
    words: List[str] = js["words"]
    entities: List[(int, int, str)] = js["ner"]
    corefs_all: Dict[str, List[(int, int)]] = js["coref"]
    n_ary_relations_all: List[Relation] = [Relation(*x)._asdict() for x in js["n_ary_relations"]]

    map_entities = {}
    new_text = []
    new_entities = []

    entities = sorted(entities, key=lambda x : x[0])
    for e in entities :
        etext = words[e[0]:e[1]]
        new_span = (len(new_text), len(new_text) + len(etext), e[-1])
        new_text += etext
        new_entities.append(list(new_span))
        map_entities[tuple([e[0], e[1]])] = (new_span[0], new_span[1])

    new_para = [(0, len(new_text))]
    new_corefs = {x:[map_entities[tuple(y)] for y in v] for x, v in corefs_all.items()}

    return {
        "sections" : new_para,
        "words" : new_text,
        "ner" : new_entities,
        "coref" : new_corefs,
        "n_ary_relations" : n_ary_relations_all
    }

span_feature_size = 13

def span_pair_features(e1, e2, w1, w2, words) :
    char_sim_features = character_similarity_features(w1, w2, max_ng=3)
    features = np.array(
        [(e1[1] - e2[0]) / len(words), (e1[1] - e1[0]) / 10, (e2[1] - e2[0]) / 10] + char_sim_features
    )

    return features
