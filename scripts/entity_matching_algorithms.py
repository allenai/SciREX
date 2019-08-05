from typing import List, Dict
from fuzzywuzzy import fuzz

import re

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))

clean_text = lambda w : re.sub(r'\s+', ' ', re.sub(r'[^\w\s\.]', ' ', w)).lower().split()
def char_sim(w1:str, w2:str, ng:int = 3, with_abbr:bool = False) :
    char1, char2 = clean_text(w1), clean_text(w2)
    ng = min(min([len(x) for x in char1 + char2]), ng)
    def get_n_grams(w_list, n) :
        n_grams = [w[i:i+n] for w in w_list for i in range(len(w)-n+1)] 
        if with_abbr :
            n_grams += ["".join([w[0] for w in w_list[i:i+n]]) for i in range(len(w_list)-n+1)]
        return n_grams
    char1, char2 = get_n_grams(char1, ng), get_n_grams(char2, ng)
    if len(char1) == 0  and len(char2) == 0 :
        print(char1, char2, clean_text(w1), clean_text(w2), ng)
    return jaccard_similarity(char1, char2)

from fuzzywuzzy import fuzz
def fuzzy_match_with_any(w1:str, w2:str) :
    return fuzz.token_sort_ratio(" ".join(clean_text(w1)), " ".join(clean_text(w2))) / 100

entity_similarity_metric = {
    'Material' : (lambda x, y : char_sim(x, y, 3, True), 0.101010),
    'Method' : (lambda x, y : char_sim(x, y, 3, False), 0.30),
    'Task' : (lambda x, y : char_sim(x, y, 3, True), 0.353535),
    'Metric' : (lambda x, y : char_sim(x, y, 3, True), 0.111111)
}

def match_entity_with_best_truth(enttype, entity, true_list) :
    sim_metric, thresh = entity_similarity_metric[enttype]
    scores = [sim_metric(entity, x) for x in true_list]
    max_score = max(scores)
    matches = [true_list[i] for i, x in enumerate(scores) if x > thresh]
    return matches

##########################################################################################################

def exact_match_in_concatenated(true_value: str, predicted_value_list: List[str]) :
    return true_value in " ".join(predicted_value_list)

def exact_match_with_any(true_value: str, predicted_value_list: List[str]) :
    return true_value in predicted_value_list

def exact_match_in_any(true_value: str, predicted_value_list: List[str]) :
    return any([true_value in x for x in predicted_value_list])

def fuzzy_match_with_any(true_value: str, predicted_value_list: List[str], min_ratio:int = 70) :
    return any([((fuzz.token_sort_ratio(true_value, x) > min_ratio) or true_value in x) for x in predicted_value_list])