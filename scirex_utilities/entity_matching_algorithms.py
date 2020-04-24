from typing import List
from fuzzywuzzy import fuzz

import re

def match_abbr(a, b) :
    if len(a.split()) > len(b.split()) :
        a, b = b, a
    it = iter(b)
    return all(c in it for c in a) * len(a)/len(b.split(' '))

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))


clean_text = lambda w: re.sub(r"\s+", " ", re.sub(r"[^\w\s\.]", " ", w)).lower().split()


def get_n_grams(w_list, n):
    n_grams = []
    for w in w_list:
        if len(w) < n:
            n_grams += [w]
        else:
            n_grams += [w[i : i + n] for i in range(len(w) - n + 1)]
    return n_grams


def get_n_grams_with_abbr(w_list, n, with_abbr=True, return_sep=False):
    n_grams = get_n_grams(w_list, n)
    if with_abbr:
        abbr_n_grams = get_n_grams(["".join([w[0] for w in w_list])], n)
        if return_sep:
            return n_grams, abbr_n_grams

        n_grams += abbr_n_grams
    return n_grams


def char_sim(w1: str, w2: str, ng: int = 3, with_abbr: bool = False) -> float:
    char1, char2 = clean_text(w1), clean_text(w2)
    if len(char1) == 0 or len(char2) == 0:
        print(char1, w1, char2, w2)
        return 0.0
    ng = min(min(max([len(x) for x in char1]), max([len(x) for x in char2])), ng)
    char1, char2 = (
        get_n_grams_with_abbr(char1, ng, with_abbr=with_abbr),
        get_n_grams_with_abbr(char2, ng, with_abbr=with_abbr),
    )
    if len(char1) == 0 and len(char2) == 0:
        print(char1, char2, clean_text(w1), clean_text(w2), ng)
    
    return max(jaccard_similarity(char1, char2), match_abbr(" ".join(clean_text(w1)), " ".join(clean_text(w2))))


def fuzzy_match_with_any(w1: str, w2: str) -> float:
    return fuzz.token_sort_ratio(" ".join(clean_text(w1)), " ".join(clean_text(w2))) / 100


entity_similarity_metric = {
    "Material": (lambda x, y: char_sim(x, y, 3, True), 0.101010),
    "Method": (lambda x, y: char_sim(x, y, 3, False), 0.31),
    "Task": (lambda x, y: char_sim(x, y, 3, True), 0.353535),
    "Metric": (lambda x, y: char_sim(x, y, 3, True), 0.111111),
}


def match_entity_with_best_truth(enttype, entity, true_list):
    sim_metric, thresh = entity_similarity_metric[enttype]
    scores = [sim_metric(entity, x) for x in true_list]
    matches = sorted([(true_list[i], x) for i, x in enumerate(scores) if x > thresh], key=lambda x : -x[1])
    if len(matches) > 0:
        assert matches[0][1] == max(scores), "Best score not first entry"
    return [x[0] for x in matches]


##########################################################################################################


def is_exact_match_in_concatenated(true_value: str, predicted_value_list: List[str]) -> bool:
    return true_value in " ".join(predicted_value_list)


def is_exact_match_with_any(true_value: str, predicted_value_list: List[str]) -> bool:
    return true_value in predicted_value_list


def is_exact_match_in_any(true_value: str, predicted_value_list: List[str]) -> bool:
    return any([true_value in x for x in predicted_value_list])


def is_fuzzy_match_with_any(true_value: str, predicted_value_list: List[str], min_ratio: int = 70) -> bool:
    return any([((fuzz.token_sort_ratio(true_value, x) > min_ratio) or true_value in x) for x in predicted_value_list])
