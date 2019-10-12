import os

import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()
import re

from collections import namedtuple

LabelSpan = namedtuple("Span", ["start", "end", "token_start", "token_end", "entity", "links"])

from scripts.analyse_pwc_entity_results import *
from scripts.entity_utils import *

# from bratreader.repomodel import RepoModel

def process_folder(folder: str):
    span_labels = {}
    map_T_to_span = {}
    if not os.path.isdir(folder) or "document.txt" not in os.listdir(folder):
        print(folder, " have not document")
        return None
    doc_file = open(os.path.join(folder, "document.txt")).read()
    ann_file = open(os.path.join(folder, "document.ann")).read().strip()
    annotations = ann_file.split("\n")
    for ann in annotations:
        ann_type, ann = ann.split("\t", 1)
        if ann_type[0] == "T":
            ann, ann_text = ann.split("\t")
            if ";" in ann:
                continue
            else:
                enttype, span_start, span_end = ann.split()
            span_start, span_end = int(span_start), int(span_end)
            if (span_start, span_end) in span_labels:
                assert "Span already present"
            else:
                span_labels[(span_start, span_end)] = {"E": enttype, "A": set(), "T": ann_text}
                map_T_to_span[ann_type] = (span_start, span_end)
        if ann_type[0] == "A":
            ann, ann_T = ann.split()
            if ann_T in map_T_to_span:
                span_labels[map_T_to_span[ann_T]]["A"].add(ann)
            else:
                print("Attribute before Trigger")
    return span_labels, doc_file


def get_all_document_annotations(brat_folder: str):
    map_id_to_ann = {}
    for f in tqdm(os.listdir(brat_folder)):
        try:
            map_id_to_ann[f] = process_folder(os.path.join(brat_folder, f))
        except Exception as e:
            print(f)
    return map_id_to_ann


def process_back_to_dataframe(span_labels, doc_text):
    sentences = doc_text.split("\n ")
    assert sentences[-1] == ""
    sentences = [x + "\n " for x in sentences[:-1]]
    sentence_limits = np.cumsum([len(x) for x in sentences])
    sentence_limits = list(zip([0] + list(sentence_limits)[:-1], sentence_limits))
    for s, e in sentence_limits:
        assert doc_text[e - 2 : e] == "\n "
        assert doc_text[s] != " "

    span_labels = list(map(lambda x: [list(x[0]), x[1]], sorted(span_labels.items(), key=lambda x: x[0][0])))
    sl_ix = 0
    map_sentence_limits_to_spans = {}
    for ss, se in sentence_limits:
        map_sentence_limits_to_spans[(ss, se)] = []
        while sl_ix < len(span_labels) and span_labels[sl_ix][0][0] >= ss and span_labels[sl_ix][0][1] <= se:
            map_sentence_limits_to_spans[(ss, se)].append(span_labels[sl_ix])
            sl_ix += 1

    spans_in_l = 0
    for k, v in map_sentence_limits_to_spans.items():
        for span, _ in v:
            assert k[0] <= span[0] and k[1] >= span[1]
            spans_in_l += 1
            assert span[1] < k[1] - 1
    assert spans_in_l == len(span_labels)

    for k, v in map_sentence_limits_to_spans.items():
        for span, _ in v:
            span[0] -= k[0]
            span[1] -= k[0]

    df = []
    for sent_id, ((ss, se), st) in enumerate(zip(sentence_limits, sentences)):
        for span, d in map_sentence_limits_to_spans[(ss, se)]:
            assert st[-2:] == "\n ", st[-2:]
            assert span[1] < len(st) - 2
            assert st[span[0] : span[1]] == d["T"] and len(d["T"]) > 0, (st[span[0] : span[1]], d["T"])
        df.append({"sentence": st, "spans": map_sentence_limits_to_spans[(ss, se)], "sentence_id": sent_id})

    df = pd.DataFrame(df)
    return df


def get_dataframe_from_folder(brat_folder):
    map_changes = get_all_document_annotations(brat_folder)

    doc_df = []
    for k in map_changes:
        if map_changes[k] is None:
            continue
        df = process_back_to_dataframe(*map_changes[k])
        df["doc_id"] = k
        doc_df.append(df)

    doc_df = pd.concat(doc_df)
    return doc_df


def overlap(span_1, span_2):
    if span_1[0] >= span_2[1] or span_2[0] >= span_1[1]:
        return False
    return True


def process_cluster(cluster):
    stats = {
        "new_spans": len([x for x in cluster if "pre" not in x[1]]),
        "old_spans": len([x for x in cluster if "pre" in x[1]]),
        "type_change": 0,
        "change_attributes": 0,
    }

    if len(cluster) == 1:
        span = cluster[0][1]
        if "pre" in cluster[0][1]:
            stats["change_attributes"] += len(span["A"] ^ span["pre"]["A"])
            stats["type_change"] = 1 if span["E"] != span["pre"]["E"] else 0
            del cluster[0][1]["pre"]

    new_spans = [x for x in cluster if "pre" not in x[1]]
    old_spans = [x for x in cluster if "pre" in x[1]]

    for nspan in new_spans:
        for ospan in old_spans:
            if overlap(ospan[0], nspan[0]):
                if nspan[1]["E"] == ospan[1]["E"]:
                    attr_span = nspan[1]["A"]
                    old_attr = ospan[1]["A"]
                    if len(attr_span) == 0:
                        nspan[1]["A"] |= old_attr
                    else:
                        if nspan[1]["A"] != old_attr:
                            stats["change_attributes"] += 1
                else:
                    stats["type_change"] += 1

    new_spans = sorted(new_spans, key=lambda x: x[0])
    for i in range(len(new_spans) - 1):
        if new_spans[i][0][1] > new_spans[i + 1][0][0]:
            breakpoint()

    new_spans = [
        LabelSpan(
            start=x[0][0], end=x[0][1], entity=x[1]["E"], links=x[1]["A"], token_start=None, token_end=None
        )._asdict()
        for x in new_spans
    ]

    stats["spans_kept"] = len(new_spans)

    return new_spans, stats


# Cases 1 : Pre entity have labels / post don't -> copy labels / delete pre entity
# Cases 2 : Pre entity have labels / post also have labels -> don't copy labels / delete pre entity
# Cases 3 : If post entity have different type than pre entity, remove pre entity
def normalize_spans(row):
    span_list_1, span_list_2 = row["spans_old"], row["spans_new"]
    map_1_span_to_ix = {tuple(k): v for k, v in span_list_1}
    if len(span_list_2) == 0:
        return [], None

    spans = [tuple(x[0]) for x in span_list_2]
    if len(spans) != len(set(spans)):
        assert "Duplicate spans", span_list_2

    span_list_2 = sorted(span_list_2, key=lambda x: x[0])
    stats = []

    j = 0
    clusters = []
    curr_cluster = []
    cstart, cend = -1, -1
    for j in range(len(span_list_2)):
        cspan = span_list_2[j]
        if tuple(cspan[0]) in map_1_span_to_ix:
            cspan[1]["pre"] = map_1_span_to_ix[tuple(cspan[0])]
        if cstart == -1:
            curr_cluster.append(cspan)
            cstart, cend = cspan[0][0], cspan[0][1]
        elif cspan[0][0] < cend:
            curr_cluster.append(cspan)
            cend = max(cend, cspan[0][1])
        else:
            curr_cluster, cluster_stats = process_cluster(curr_cluster)
            stats.append(cluster_stats)
            clusters.append(curr_cluster)
            curr_cluster = [cspan]
            cstart, cend = cspan[0][0], cspan[0][1]

    curr_cluster, cluster_stats = process_cluster(curr_cluster)
    stats.append(cluster_stats)
    clusters.append(curr_cluster)

    clusters = sorted([z for x in clusters for z in x], key=lambda x: (x["start"], x["end"]))
    for i in range(len(clusters) - 1):
        if clusters[i]["end"] > clusters[i + 1]["start"]:
            breakpoint()

    stats_reduced = {}
    for s in stats:
        for k, v in s.items():
            if k not in stats_reduced:
                stats_reduced[k] = v
            else:
                stats_reduced[k] += v
    return clusters, stats_reduced


def add_token_index(row):
    sentence = row["sentence_old"]
    words = sentence.split()
    word_lens = np.cumsum([len(w) + 1 for w in words])
    ends = list(word_lens - 1)
    starts = [0] + list(word_lens[:-1])

    for i, (start, end) in enumerate(zip(starts, ends)):
        assert sentence[start:end] == words[i]

    new_cluster = []
    cluster = row["cluster"]
    for i, span in enumerate(cluster):
        if not (span["start"] in starts):
            if sentence[span["start"]] == "":
                span["start"] += 1
            else:
                span["start"] = min(starts, key=lambda x: abs(x - span["start"]) if x < span["start"] else float("inf"))

        if not (span["end"] in ends):
            if sentence[span["end"] - 1] == "":
                span["end"] -= 1
            else:
                span["end"] = min(ends, key=lambda x: abs(x - span["end"]) if x > span["end"] else float("inf"))

        span["token_start"] = starts.index(span["start"])
        span["token_end"] = ends.index(span["end"]) + 1
        new_cluster.append(LabelSpan(**span))

    return words, new_cluster


def compare_brat_annotations(ann_old_df, ann_new_df):
    df_merged = ann_old_df.merge(ann_new_df, on=["doc_id", "sentence_id"], suffixes=("_old", "_new"))
    output = df_merged.progress_apply(normalize_spans, axis=1)
    df_merged["cluster"], df_merged["stats"] = list(zip(*output))
    output = df_merged.progress_apply(add_token_index, axis=1)
    df_merged["words"], df_merged["entities"] = list(zip(*output))

    df_merged = (
        df_merged.sort_values(["doc_id", "sentence_id"])
        .reset_index(drop=True)
        .drop(columns=["spans_old", "spans_new", "sentence_new", "cluster"])
        .rename(columns={"sentence_old": "sentence"})
    )

    return df_merged


def generate_relations_in_pwc_df(pwc_df):
    pwc_df_keep = pwc_df[["s2_paper_id"] + true_entities + ["score"]].rename(columns=map_true_entity_to_available)
    pwc_df_keep = (
        pwc_df_keep[(~pwc_df_keep.duplicated()) & (pwc_df_keep.s2_paper_id != "not_found")]
        .sort_values(["s2_paper_id"] + used_entities + ["score"])
        .reset_index(drop=True)
    )

    pwc_df_keep[used_entities] = pwc_df_keep[used_entities].applymap(lambda x: re.sub(r"[^\w-]", "_", x))
    pwc_df_keep = (
        pwc_df_keep.groupby("s2_paper_id")
        .apply(lambda x: list(x[used_entities + ["score"]].itertuples(index=False, name="Relation")))
        .reset_index()
        .rename(columns={0: "Relations"})
    )

    def convert_to_index(tuples):
        index_map = {k: {} for k in used_entities}
        for i, t in enumerate(tuples):
            t = t._asdict()
            for k in used_entities:
                if t[k] in index_map[k]:
                    index_map[k][t[k]].add(i)
                else:
                    index_map[k][t[k]] = set([i])

        return index_map

    inverse_relations = pwc_df_keep["Relations"].progress_apply(convert_to_index)
    pwc_df_relations = pd.concat([pwc_df_keep, pd.DataFrame.from_records(inverse_relations)], axis=1)
    return pwc_df_relations


def combine_brat_to_original_data(
    pwc_doc_file, pwc_sentence_file, pwc_prediction_file, original_brat_anno_folder, annotated_brat_anno_folder
):
    pwc_df = load_pwc_full_text(pwc_doc_file)
    pwc_grouped = (
        pwc_df.groupby("s2_paper_id")[["dataset", "task", "model_name", "metric"]]
        .aggregate(lambda x: list(set(tuple(x))))
        .reset_index()
    )

    pwc_sentences = load_pwc_sentence_predictions(pwc_sentence_file, pwc_prediction_file)

    pwc_sentences = pwc_sentences.merge(pwc_grouped, left_on="doc_id", right_on="s2_paper_id")
    pwc_sentences = pwc_sentences.sort_values(by=["doc_id", "section_id", "para_id", "sentence_id"]).reset_index(
        drop=True
    )

    df_changed = get_dataframe_from_folder(annotated_brat_anno_folder)
    df_original = get_dataframe_from_folder(original_brat_anno_folder)

    df_merged = compare_brat_annotations(df_original, df_changed)

    assert (pwc_sentences["words"].apply(lambda words: " ".join(words + ["\n "])) != df_merged["sentence"]).sum() == 0

    pwc_df_relations = generate_relations_in_pwc_df(pwc_df)

    df_concat = pd.concat(
        [
            pwc_sentences[["doc_id", "para_id", "section_id", "sentence_id"]],
            df_merged[["sentence", "words", "entities", "stats"]],
        ],
        axis=1,
    )

    df_concat = (
        df_concat.merge(pwc_df_relations, right_on="s2_paper_id", left_on="doc_id", suffixes=("", "_Rel"))
        .drop("s2_paper_id", axis=1)
        .rename(columns={k: k + "_Rel" for k in used_entities})
    )

    def remove_incorrect_links(row):
        index_map = {k: set(row[k + "_Rel"].keys()) for k in used_entities}
        new_spans = []
        spans = row["entities"]
        for span in spans:
            span = span._asdict()
            span["links"] &= index_map[span["entity"]]
            new_spans.append(LabelSpan(**span))
        return new_spans

    df_concat["entities"] = df_concat.progress_apply(remove_incorrect_links, axis=1)

    df_concat["para_num"] = None
    df_concat["sentence_num"] = None

    def add_nums(rows, columns, name):
        rows[name] = list(rows.groupby(columns).grouper.group_info[0])
        return rows

    df_concat = df_concat.groupby("doc_id").progress_apply(lambda x: add_nums(x, ["section_id", "para_id"], "para_num"))
    df_concat = df_concat.groupby("doc_id").progress_apply(
        lambda x: add_nums(x, ["section_id", "para_id", "sentence_id"], "sentence_num")
    )

    return df_concat
