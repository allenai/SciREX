from typing import List, Tuple
from scirex.data.utils.span_utils import is_x_in_y

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

def extract_sentence_features(sentences, words, entities):
    entities_to_features_map = {}
    sentence_features = [get_features_for_sections(sents, words) for sents in sentences]
    for e in entities:
        index = [
            (i, j)
            for i, sents in enumerate(sentences)
            for j, sspan in enumerate(sents)
            if is_x_in_y(e, sspan)
        ]
        assert len(index) == 1, breakpoint()

        i, j = index[0]
        entities_to_features_map[(e[0], e[1])] = sentence_features[i][j]

    return entities_to_features_map


def filter_to_doctaet(json_dict):
    sentences = json_dict["sentences"]
    words = json_dict["words"]
    sections = json_dict["sections"]

    section_features = get_features_for_sections(sections, words)

    new_words = []
    experiment_words = []
    for sents, sec_fs in zip(sentences, section_features):
        if "Heading" in sec_fs or "Abstract" in sec_fs:
            # Check if section is Abstract or Heading
            for s in sents :
                new_words += words[s[0]:s[1]]
        else:
            features = get_features_for_sections(sents, words)
            # Check if sentence include any IBM labels
            for s, f in zip(sents, features) :
                if 'Dataset' in f or 'Experiment' in f :
                    experiment_words += words[s[0]:s[1]]

    new_words += experiment_words[:150]

    return new_words