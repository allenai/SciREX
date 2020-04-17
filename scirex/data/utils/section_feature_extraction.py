from collections import defaultdict
from copy import deepcopy
from typing import Callable, Dict, List, Tuple

import numpy as np

is_x_in_y = lambda x, y: x[0] >= y[0] and x[1] <= y[1]

experiment_words_to_check = set("experiment|evaluation|evaluate|evaluate".split("|"))
dataset_words_to_check = set("dataset|corpus|corpora".split("|"))


def get_features_for_sections(sections: List[Tuple[int, int]], words_list: List[str]):
    features_list = []
    for i, (s, e) in enumerate(sections):
        features = []
        words = " ".join(words_list[s:e]).lower()
        if i == 0:
            features.append("Heading")

        if "abstract" in words:
            features.append("Abstract")
        if "introduction" in words:
            features.append("Introduction")

        if any(w in words for w in dataset_words_to_check):
            features.append("Dataset")
            features.append("Experiment")

        if any(w in words for w in experiment_words_to_check):
            features.append("Experiment")

        features_list.append(sorted(list(set(features))))

    return features_list


def filter_sentences(
    sentences: List[List[Tuple[int, int]]],
    keep_sentence: List[List[bool]],
    words: List[str],
    entities: Dict[Tuple[int, int], str],
):
    map_old_spans_to_new = {}
    new_words = []
    new_sections = []
    new_sentences = []

    entity_in_atleast_one_sentence = defaultdict(int)

    for sents, keep_sents in zip(sentences, keep_sentence):
        section_start = len(new_words)
        section_sentences = []
        for sent, keep_sent in zip(sents, keep_sents):
            old_sent_entities = [e for e in entities if is_x_in_y(e, sent)]
            for e in old_sent_entities:
                entity_in_atleast_one_sentence[e] += 1

            if keep_sent:
                old_sent_start = sent[0]
                new_sent_start = len(new_words)
                diff = new_sent_start - old_sent_start
                sent_words = words[sent[0] : sent[1]]

                new_sent_entities = [(diff + e[0], diff + e[1]) for e in old_sent_entities]

                new_words += sent_words
                for o, n in zip(old_sent_entities, new_sent_entities):
                    map_old_spans_to_new[o] = n

                section_sentences.append((new_sent_start, len(new_words)))

        section_end = len(new_words)
        if section_end > section_start:
            new_sections.append((section_start, section_end))
            new_sentences.append(section_sentences)

    for e in entities:
        assert entity_in_atleast_one_sentence[e] == 1, breakpoint()

    return new_sections, new_words, new_sentences, map_old_spans_to_new


def filter_json_dict(json_dict, keep_sentence: List[List[bool]]):
    new_sections, new_words, new_sentences, map_old_spans_to_new = filter_sentences(
        json_dict["sentences"], keep_sentence, json_dict["words"], json_dict["ner"]
    )

    new_json_dict = deepcopy(json_dict)
    new_json_dict["words"] = new_words
    new_json_dict["sections"] = new_sections
    new_json_dict["sentences"] = new_sentences
    new_json_dict["ner"] = {
        map_old_spans_to_new[k]: v for k, v in json_dict["ner"].items() if k in map_old_spans_to_new
    }
    new_json_dict["coref"] = {
        k: [map_old_spans_to_new[tuple(x)] for x in v if tuple(x) in map_old_spans_to_new]
        for k, v in new_json_dict["coref"].items()
    }

    map_new_span_to_old = {v: k for k, v in map_old_spans_to_new.items()}
    for k in new_json_dict["ner"]:
        assert (
            new_json_dict["words"][k[0] : k[1]]
            == json_dict["words"][map_new_span_to_old[k][0] : map_new_span_to_old[k][1]]
        )

    return new_json_dict


def filter_to_abstract(json_dict):
    sentences = json_dict["sentences"]
    sections = json_dict["sections"]
    words = json_dict["words"]

    section_features = get_features_for_sections(sections, words)

    keep_sentence = []
    for sec_f, sents in zip(section_features, sentences):
        keep = "Heading" in sec_f or "Abstract" in sec_f
        keep_sentence.append([keep for _ in range(len(sents))])

    return filter_json_dict(json_dict, keep_sentence)


def filter_to_doctaet(json_dict):
    sentences = json_dict["sentences"]
    words = json_dict["words"]
    sections = json_dict["sections"]

    section_features = get_features_for_sections(sections, words)

    keep_sentence = []
    for sents, sec_fs in zip(sentences, section_features):
        if "Heading" in sec_fs or "Abstract" in sec_fs:
            # Check if section is Abstract or Heading
            ks = [True for _ in range(len(sents))]
        else:
            features = get_features_for_sections(sents, words)
            # Check if sentence include any IBM labels
            ks = [len(fs) > 0 for fs in features]

        keep_sentence.append(ks)

    return filter_json_dict(json_dict, keep_sentence)


filter_dispatcher: Dict[str, Callable] = {
    "abstract": filter_to_abstract,
    "doctaet": filter_to_doctaet,
    "full": lambda json_dict: json_dict,
}