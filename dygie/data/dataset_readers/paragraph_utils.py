import numpy as np
from dygie.data.dataset_readers.read_pwc_dataset import is_x_in_y

import intervals as I


def collapse_paragraphs(plist, min_len=100, max_len=400):
    para_lengths = [p[1] - p[0] for p in plist]
    new_paragraphs = []
    cur_para_len = 0
    for p in para_lengths:
        if cur_para_len > max_len:
            new_paragraphs.append(cur_para_len)
            cur_para_len = 0
        if p < min_len:
            cur_para_len += p
        else:
            new_paragraphs.append(cur_para_len + p)
            cur_para_len = 0

    new_paragraphs.append(cur_para_len)
    assert sum(para_lengths) == sum(new_paragraphs), (sum(para_lengths), sum(new_paragraphs))
    return new_paragraphs


def break_paragraphs(plist, max_len=400):
    new_paragraphs = []
    for p in plist:
        if p < max_len:
            new_paragraphs.append(p)
        else:
            new_paragraphs += [max_len] * (p // max_len)
            if p % max_len > 0:
                new_paragraphs.append(p % max_len)

    assert sum(plist) == sum(new_paragraphs), (sum(plist), sum(new_paragraphs))
    return new_paragraphs


def move_boundaries(plist, elist):
    ends = np.cumsum(plist)
    starts = ends - np.array(plist)
    starts, ends = list(starts), list(ends)

    elist = sorted(elist, key=lambda x: (x[0], x[1]))
    para_stack = list(zip(starts, ends))
    new_paragraphs = []
    eix = 0
    while len(para_stack) > 0:
        p = para_stack.pop(0)

        while True:
            if eix >= len(elist):
                new_paragraphs.append(p)
                break
            elif elist[eix][0] >= p[0] and elist[eix][1] <= p[1]:
                eix += 1
            elif elist[eix][0] >= p[1]:
                new_paragraphs.append(p)
                break
            elif elist[eix][0] >= p[0]:
                p1 = para_stack.pop(0)
                new_paragraphs.append((p[0], elist[eix][1]))
                para_stack.insert(0, (elist[eix][1], p1[1]))
                eix += 1
                break

    assert new_paragraphs[0][0] == starts[0]
    assert new_paragraphs[-1][1] == ends[-1]
    for p, q in zip(new_paragraphs[:-1], new_paragraphs[1:]):
        assert p[1] == q[0]

    for e in elist:
        done = False
        for p in new_paragraphs:
            if is_x_in_y((e[0], e[1]), p):
                done = True
        assert done

    return new_paragraphs


def get_wastage(para_lengths):
    total_padded_input = [max(p) * len(p) for i, p in enumerate(para_lengths)]
    total_input = [sum(p) for i, p in enumerate(para_lengths)]
    return (sum(total_padded_input) - sum(total_input)) / sum(total_padded_input) * 100


def gen_lens(plist):
    return [p[1] - p[0] for p in plist]


experiment_words_to_check = set(
    "experiment|experiments|evaluation|evaluations".split("|")
    + "experiment|experiments|evaluation|evaluations".split("|")
    + "experiment on|experiment in|evaluation|evalualtions|evaluate|evaluated".split("|")
)

dataset_words_to_check = set("dataset|datasets|corpus|corpora".split("|"))


def get_features_for_sections(sections, words_list):
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

        for w in dataset_words_to_check:
            if w in words:
                features.append("Dataset")
                features.append("Experiment")
                break

        for w in experiment_words_to_check:
            if w in words:
                features.append("Experiment")
                break

        features_list.append(sorted(list(set(features))))

    return features_list


def subset_sections_to_important(sections, words):
    keep = [i for i, x in enumerate(get_features_for_sections(sections, words)) if len(x) > 0]
    return sorted(list(set(keep)))
