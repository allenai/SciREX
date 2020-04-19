from scirex_utilities.entity_matching_algorithms import *
from scirex_utilities.entity_utils import *
import numpy as np 

def match_clusters(c1, c2) :
    scores = np.zeros((len(c1), len(c2)))
    for i in range(len(c1)) :
        for j in range(len(c2)) :
            scores[i, j] = char_sim(c1[i], c2[j], 3)

    return scores

def generate_thresholds(data) : 
    th = {k:1.0 for k in used_entities}
    return th