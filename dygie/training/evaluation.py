from scripts.entity_matching_algorithms import *
from scripts.entity_utils import *
from collections import Counter
import numpy as np

def get_reference_and_gold(d) :
    span_map = {(x[0], x[1]):x[2].split('_')[0] for x in d['ner']}
    corefs = {k.replace('_', ' '):[" ".join(d['words'][x[0]:x[1]]) for x in v] for k, v in d['coref'].items()}
    coref_type = {k.replace('_', ' '):Counter([span_map[(x[0], x[1])] for x in v]).most_common(1) for k, v in d['coref'].items()}
    return corefs, coref_type

def generate_thresholds(dev_data) :
    scores = {x:[] for x in used_entities}
    for d in dev_data :
        dk, dc = get_reference_and_gold(d)
        ds = {}
        for k in dk :
            s = [0.0]
            for v in dk[k] :
                score = char_sim(k, v, 3)
                s.append(score)
            ds[k] = max(s)

        for k in ds :
            if len(dc[k]) > 0:
                scores[dc[k][0][0]].append(ds[k])
                
    for k, s in scores.items() :
        scores[k] = np.percentile(s, 75)
                
    return scores

def match_clusters(c1, c2) :
    scores = np.zeros((len(c1), len(c2)))
    for i, w1 in enumerate(c1) :
        for j, w2 in enumerate(c2) :
            scores[i, j] = char_sim(w1, w2, 3)

    return scores