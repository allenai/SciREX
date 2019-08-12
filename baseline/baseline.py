from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm

from scripts.entity_matching_algorithms import *
from collections import namedtuple

MentionTuple = namedtuple(
    "Mention",
    "start,end,token_start,token_end,entity,links,sentence_id,section_id,para_id"
    + ",mention_pos_in_sentence,mention_pos_in_paragraph,mention_pos_in_document"
    + ",word_pos_in_sentence,word_pos_in_paragraph,word_pos_in_document,n_tokens,n_chars,text",
)

Mention = namedtuple(
    "Mention","start,end,text,mention_pos"
)

def character_similarity_features(span_1: MentionTuple, span_2: MentionTuple, max_ng: int = 3) -> Dict[str, float]:
    w1, w2 = span_1.text, span_2.text
    char1, char2 = clean_text(w1), clean_text(w2)
    if len(char1) == 0 or len(char2) == 0:
        return {}

    ng = min(min(max([len(x) for x in char1]), max([len(x) for x in char2])), max_ng)
    features = {}
    for n in range(1, ng + 1):
        cgrams_1, cgrams_abbr_1 = get_n_grams_with_abbr(char1, n, return_sep=True)
        cgrams_2, cgrams_abbr_2 = get_n_grams_with_abbr(char2, n, return_sep=True)
        features["word_word_sim_" + str(n)] = jaccard_similarity(cgrams_1, cgrams_2)
        features["word_abbr_sim_" + str(n)] = max(
            jaccard_similarity(cgrams_1, cgrams_abbr_2), jaccard_similarity(cgrams_2, cgrams_abbr_1)
        )
        features["abbr_abbr_sim_" + str(n)] = jaccard_similarity(cgrams_abbr_1, cgrams_abbr_2)

    features["fuzzy_word_word_sim"] = fuzzy_match_with_any(w1, w2)
    return features


def words_between_spans(span_1: MentionTuple, span_2: MentionTuple):
    features = {}
    features["word_distance_in_document"] = abs(span_1.start - span_2.start)
    return features


def mention_between_spans(span_1: MentionTuple, span_2: MentionTuple):
    features = {}
    features["mention_distance_in_document"] = abs(span_1.mention_pos - span_2.mention_pos)
    return features


def get_all_span_pair_features(span_1: MentionTuple, span_2: MentionTuple):
    features = {}
    features.update(words_between_spans(span_1, span_2))
    features.update(mention_between_spans(span_1, span_2))
    features.update(character_similarity_features(span_1, span_2))
    return features


def add_features_to_row(row, info):
    utils = {}
    sentence_id = row["name"]
    for column in ["entities", "words"]:
        utils["n_" + column] = info["n_" + column][sentence_id]
        utils["start_" + column] = info["prev_" + column][sentence_id] - utils["n_" + column]
        utils["total_" + column] = info["total_" + column]
        prev_para_id = row["para_num"] - 1
        utils["para_start_" + column] = utils["start_" + column] - (
            info["para_prev_" + column][prev_para_id] if prev_para_id >= 0 else 0
        )
        utils["para_total_" + column] = info["para_length_" + column][row["para_num"]]

    new_clusters = []
    entities = row["entities"]
    sentence = row["sentence"]

    assert len(entities) == utils["n_entities"]
    for i, span in enumerate(entities):
        new_span = {k: v for k, v in span._asdict().items()}
        new_span["sentence_id"] = row["sentence_num"]
        new_span["section_id"] = row["section_id"]
        new_span["para_id"] = row["para_num"]

        new_span["mention_pos_in_sentence"] = (i + 1) / utils["n_entities"]
        new_span["mention_pos_in_paragraph"] = (utils["para_start_entities"] + i + 1) / utils["para_total_entities"]
        new_span["mention_pos_in_document"] = (utils["start_entities"] + i + 1) / utils["total_entities"]

        new_span["word_pos_in_sentence"] = (span.token_start + 1) / utils["n_words"]
        new_span["word_pos_in_paragraph"] = (utils["para_start_words"] + span.token_start + 1) / utils[
            "para_total_words"
        ]
        new_span["word_pos_in_document"] = (utils["start_words"] + span.token_start + 1) / utils["total_words"]

        new_span["n_tokens"] = span.token_end - span.token_start
        new_span["n_chars"] = span.end - span.start
        new_span["text"] = sentence[span.start : span.end]
        new_span = MentionTuple(**new_span)
        new_clusters.append(new_span)

    return new_clusters


def compute_features(rows):
    info = {}

    for column in ["entities", "words"]:
        data = rows[column].values
        data_lens = [len(x) for x in data]
        cumsums = np.cumsum(data_lens)
        total_data = sum(data_lens)
        info["prev_" + column] = cumsums
        info["total_" + column] = total_data
        info["n_" + column] = data_lens

    para_ids = rows["para_num"].values
    rows = rows.to_dict("records")

    fclusters = [None] * len(rows)
    for column in ["entities", "words"]:
        para_lengths = [0] * (np.max(para_ids) + 1)
        for i, row in enumerate(rows):
            para_lengths[row["para_num"]] += len(row[column])

        info["para_length_" + column] = para_lengths
        info["para_prev_" + column] = np.cumsum(para_lengths)

    for i, row in enumerate((rows)):
        row["name"] = i
        fclusters[i] = add_features_to_row(row, info)

    return fclusters


