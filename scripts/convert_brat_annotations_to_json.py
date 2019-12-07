import json
import logging
import os
import re
from collections import namedtuple
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import spacy
from scripts.analyse_pwc_entity_results import *
from scripts.entity_utils import *
from spacy.tokens import Doc
from tqdm import tqdm

tqdm.pandas()
LabelSpan = namedtuple("Span", 
["start", "end", "token_start", "token_end", "entity", "links", "modified"])
logging.basicConfig(level=logging.INFO)


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


nlp = spacy.load("en")
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)


def process_folder(folder: str) -> Tuple[dict, str]:
    span_labels = {}
    map_T_to_span = {}
    if not os.path.isdir(folder) or "document.txt" not in os.listdir(folder):
        print(folder, " have not document")
        return None
    doc_text = open(os.path.join(folder, "document.txt")).read()
    ann_file = open(os.path.join(folder, "document.ann")).read().strip()

    annotations = [x.split("\t", 1) for x in ann_file.split("\n")]
    annotations = sorted(annotations, key=lambda x: 0 if x[0] == "T" else 1)
    for ann_type, ann in annotations:
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
    return span_labels, doc_text


def get_all_document_annotations(brat_folder: str) -> Dict[str, Tuple[dict, str]]:
    map_id_to_ann = {}
    for f in tqdm(os.listdir(brat_folder)):
        try:
            map_id_to_ann[f] = process_folder(os.path.join(brat_folder, f))
        except Exception as e:
            print(f)
    return map_id_to_ann


def process_back_to_dataframe(span_labels: Dict[Tuple[int, int], dict], doc_text: str):
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

    assert df[4]["sentence"].strip() == "", breakpoint()
    df = df[5:]
    df = pd.DataFrame(df)

    return df


def get_dataframe_from_folder(brat_folder):
    logging.info("Generating DataFrame ...")
    map_changes = get_all_document_annotations(brat_folder)

    logging.info("Done generating DataFrame")
    doc_df = []
    for k in tqdm(map_changes):
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

    old_spans = [x for x in cluster if "pre" in x[1]]
    new_spans = [x for x in cluster if "pre" not in x[1]]

    old_spans_modified, old_spans_unmodified = [], []
    for span, info in old_spans:
        if [info[k] for k in ["E", "T", "A"]] == [info["pre"][k] for k in ["E", "T", "A"]]:
            del info["pre"]
            if any(overlap(span, n_span) for n_span, _ in new_spans):
                continue
            old_spans_unmodified.append((span, info))
        else:
            del info["pre"]
            if any(overlap(span, n_span) for n_span, _ in new_spans):
                continue
            old_spans_modified.append((span, info))

    assert all((si == sj or not overlap(si[0], sj[0])) for si in new_spans for sj in new_spans), breakpoint()
    assert len(old_spans_unmodified) == 0 or len(old_spans_modified) == 0, breakpoint()
    assert all((not overlap(ospan, nspan)) for ospan, _ in old_spans_modified for nspan, _ in new_spans), breakpoint()
    assert all((not overlap(ospan, nspan)) for ospan, _ in old_spans_unmodified for nspan, _ in new_spans), breakpoint()

    if len(old_spans_modified + old_spans_unmodified) > 0 and len(new_spans) > 0:
        breakpoint()

    new_spans = [
        LabelSpan(
            start=x[0][0],
            end=x[0][1],
            entity=x[1]["E"],
            links=x[1]["A"],
            token_start=None,
            token_end=None,
            modified=True,
        )._asdict()
        for x in new_spans + old_spans_modified
    ]

    new_spans += [
        LabelSpan(
            start=x[0][0],
            end=x[0][1],
            entity=x[1]["E"],
            links=x[1]["A"],
            token_start=None,
            token_end=None,
            modified=False,
        )._asdict()
        for x in old_spans_unmodified
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

    clusters = []
    curr_cluster = []
    cstart, cend = -1, -1
    for (start, end), span_info in span_list_2:
        cspan = ((start, end), span_info)
        if (start, end) in map_1_span_to_ix:
            span_info["pre"] = map_1_span_to_ix[(start, end)]
        if cstart == -1:  # (Start First Cluster)
            curr_cluster.append(cspan)
            cstart, cend = start, end
        elif start < cend:  # Append to current cluster
            curr_cluster.append(cspan)
            cend = max(cend, end)
        else:  # Start new cluster
            curr_cluster, cluster_stats = process_cluster(curr_cluster)
            stats.append(cluster_stats)
            clusters.append(curr_cluster)
            curr_cluster = [cspan]
            cstart, cend = start, end

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
    if len(row['cluster']) == 0 :
        return []
    sentence = row["sentence_old"]
    words = row["words"]
    word_indices = row['word_indices']
    sentence_start = row["sentence_start"]
    starts, ends = list(zip(*word_indices))

    for i, (start, end) in enumerate(zip(starts, ends)):
        assert sentence[start:end] == words[i], breakpoint()

    new_cluster = []
    cluster = row["cluster"]
    for i, span in enumerate(cluster):
        assert 'start' in span, breakpoint()
        assert 'end' in span, breakpoint()
        if not (span["start"] in starts):
            if sentence[span["start"]].strip() == "":
                span["start"] += 1
            else:
                span["start"] = min(starts, key=lambda x: abs(x - span["start"]) if x < span["start"] else float("inf"))

        if not (span["end"] in ends):
            if sentence[span["end"] - 1].strip() == "":
                span["end"] -= 1
            else:
                span["end"] = min(ends, key=lambda x: abs(x - span["end"]) if x > span["end"] else float("inf"))

        span["token_start"] = starts.index(span["start"]) + sentence_start - len(words)
        span["token_end"] = ends.index(span["end"]) + 1 + sentence_start - len(words)

        for cleaned_span in new_cluster:
            if overlap(
                (span["token_start"], span["token_end"]), (cleaned_span["token_start"], cleaned_span["token_end"])
            ):
                print(row["doc_id"])
                print(" ".join(row["words"]))
                print("=" * 20)
        new_cluster.append(span)

    return new_cluster


def generate_token_and_indices(sentence):
    words = sorted(
        [(m.group(0), (m.start(), m.end())) for m in re.finditer(r"[^\s\+\-/\(\)&\[\],]+", sentence)]
        + [(m.group(0), (m.start(), m.end())) for m in re.finditer(r"[\+\-/\(\)&\[\],]+", sentence)]
        + [(m.group(0), (m.start(), m.end())) for m in re.finditer(r"\s+", sentence)],
        key=lambda x: x[1],
    )

    if len(words) == 0 or sentence.strip() == '':
        return [], []

    try :
        words, indices = list(zip(*[(t, i) for t, i in words if t.strip() != '']))
    except :
        breakpoint()

    return words, indices


def compare_brat_annotations(ann_old_df, ann_new_df):
    df_merged = ann_old_df.merge(ann_new_df, on=["doc_id", "sentence_id"], suffixes=("_old", "_new"))
    logging.info("Applying Normalize Spans ...")
    output = df_merged.progress_apply(normalize_spans, axis=1)
    df_merged["cluster"], df_merged["stats"] = list(zip(*output))

    df_merged = df_merged.sort_values(["doc_id", "sentence_id"]).reset_index(drop=True)

    logging.info("Applying Add Token Index ...")
    df_merged["words"], df_merged['word_indices'] = list(zip(*df_merged["sentence_old"].progress_apply(generate_token_and_indices)))
    df_merged["num_words"] = df_merged["words"].progress_apply(len)
    df_merged["sentence_start"] = df_merged.groupby("doc_id")["num_words"].cumsum()

    df_merged["entities"] = df_merged.apply(add_token_index, axis=1)

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

    # pwc_df_keep[used_entities] = pwc_df_keep[used_entities].applymap(lambda x: re.sub(r"[^\w-]", "_", x))
    pwc_df_keep = (
        pwc_df_keep.groupby("s2_paper_id")
        .apply(lambda x: list(x[used_entities + ["score"]].itertuples(index=False, name="Relation")))
        .reset_index()
        .rename(columns={0: "Relations"})
    )

    return pwc_df_keep


def combine_brat_to_original_data(
    pwc_doc_file, pwc_sentence_file, pwc_prediction_file, original_brat_anno_folder, annotated_brat_anno_folder
):
    logging.info("Loading pwc docs ... ")
    pwc_df = load_pwc_full_text(pwc_doc_file)
    pwc_grouped = (
        pwc_df.groupby("s2_paper_id")[["dataset", "task", "model_name", "metric"]]
        .aggregate(lambda x: list(set(tuple(x))))
        .reset_index()
    )

    pwc_df_relations = generate_relations_in_pwc_df(pwc_df)
    pwc_df_relations = pwc_df_relations.rename(columns={"s2_paper_id": "doc_id"})[["doc_id", "Relations"]]
    pwc_df_relations.index = pwc_df_relations.doc_id
    pwc_df_relations = pwc_df_relations.drop(columns=["doc_id"])
    pwc_df_relations: Dict[str, Relation] = pwc_df_relations.to_dict()["Relations"]

    method_breaks = {d:
        {   
            clean_name(rel.Method): [(i, clean_name(x)) for i, x in chunk_string(rel.Method)]
            for rel in relations
        }
        for d, relations in pwc_df_relations.items()
    }

    pwc_df_relations = {
        d: [{
                k:clean_name(x) if k != 'score' else x
                for k, x in rel._asdict().items()
                } 
            for rel in relations]
        for d, relations in pwc_df_relations.items()
    }

    logging.info("Loading PwC Sentence Predictions ... ")
    pwc_sentences = load_pwc_sentence_predictions(pwc_sentence_file, pwc_prediction_file)

    pwc_sentences = pwc_sentences.merge(pwc_grouped, left_on="doc_id", right_on="s2_paper_id")
    pwc_sentences = pwc_sentences.sort_values(by=["doc_id", "section_id", "para_id", "sentence_id"]).reset_index(
        drop=True
    )

    pwc_sentences['words'] = pwc_sentences['words'].progress_apply(lambda x : generate_token_and_indices(" ".join(x))[0])

    df_changed = get_dataframe_from_folder(annotated_brat_anno_folder)
    df_original = get_dataframe_from_folder(original_brat_anno_folder)

    df_merged = compare_brat_annotations(df_original, df_changed)

    assert (
        pwc_sentences.groupby("doc_id")["words"].agg(lambda words: [x for y in words for x in y])
        != df_merged.groupby("doc_id")["words"].agg(lambda words: [x for y in words for x in y])
    ).sum() == 0, breakpoint()

    def add_nums(rows, columns, name):
        rows[name] = list(rows.groupby(columns).grouper.group_info[0])
        return rows

    pwc_sentences["para_num"] = None
    pwc_sentences["sentence_num"] = None

    pwc_sentences = pwc_sentences.groupby("doc_id").progress_apply(
        lambda x: add_nums(x, ["section_id", "para_id"], "para_num")
    )
    pwc_sentences = pwc_sentences.groupby("doc_id").progress_apply(
        lambda x: add_nums(x, ["section_id", "para_id", "sentence_id"], "sentence_num")
    )

    words: Dict[str, List[str]] = pwc_sentences.groupby("doc_id")["words"].agg(
        lambda words: [x for y in words for x in y]
    ).to_dict()
    pwc_sentences["num_words"] = pwc_sentences["words"].apply(len)
    sentences = pwc_sentences.groupby(["doc_id", "sentence_num"])["num_words"].agg(sum)
    sections = pwc_sentences.groupby(["doc_id", "section_id"])["num_words"].agg(sum)

    sections: Dict[str, Dict[int, int]] = {level: sections.xs(level).to_dict() for level in sections.index.levels[0]}
    sentences: Dict[str, Dict[int, int]] = {level: sentences.xs(level).to_dict() for level in sentences.index.levels[0]}

    words_merged = df_merged.groupby("doc_id")["words"].agg(lambda words: [x for y in words for x in y]).to_dict()
    entities = df_merged.groupby("doc_id")["entities"].agg(lambda ents: [x for y in ents for x in y]).to_dict()

    def compute_start_end(cards):
        ends = list(np.cumsum(cards))
        starts = [0] + ends
        return list(zip([int(x) for x in starts], [int(x) for x in ends]))

    combined_information = {}
    for d in words:
        assert words[d] == words_merged[d], breakpoint()
        assert list(sentences[d].keys()) == list(range(max(sentences[d].keys()) + 1)), breakpoint()
        assert list(sections[d].keys()) == list(range(max(sections[d].keys()) + 1)), breakpoint()

        sent = compute_start_end([sentences[d][i] for i in range(len(sentences[d]))])
        sec = compute_start_end([sections[d][i] for i in range(len(sections[d]))])

        for e in entities[d] :
            del e['start']
            del e['end']

        combined_information[d] = {
            "words": words[d],
            "sentences": sent,
            "sections": sec,
            "relations": pwc_df_relations[d],
            "entities": entities[d],
            "doc_id": d,
            "method_subrelations" : method_breaks[d]
        }

    return combined_information


def _annotation_to_dict(dc):
    # convenience method
    if isinstance(dc, dict):
        ret = dict()
        for k, v in dc.items():
            k = _annotation_to_dict(k)
            v = _annotation_to_dict(v)
            ret[k] = v
        return ret
    elif isinstance(dc, str):
        return dc
    elif isinstance(dc, (set, frozenset, list, tuple)):
        ret = []
        for x in dc:
            ret.append(_annotation_to_dict(x))
        return tuple(ret)
    else:
        return dc


def annotations_to_jsonl(annotations, output_file):
    with open(output_file, "w") as of:
        for ann in sorted(annotations, key=lambda x: x["doc_id"]):
            as_json = _annotation_to_dict(ann)
            as_str = json.dumps(as_json, sort_keys=True)
            of.write(as_str)
            of.write("\n")


def propagate_annotations(data_dict: Dict[str, Any]):
    words = data_dict["words"]
    entities = data_dict["entities"]
    entities = {(e["token_start"], e["token_end"]): e for e in entities}
    assert not any(e != f and overlap(e, f) for e in entities for f in entities), breakpoint()

    new_entities = {}

    for (s, e) in entities:
        if entities[(s, e)]["modified"] == True:
            span_text = words[s:e]
            possible_matches = [
                (i, i + len(span_text)) for i in range(len(words)) if words[i : i + len(span_text)] == span_text
            ]

            for match in possible_matches:
                add_match = False
                if match in entities:
                    if entities[match].get("proped", False):
                        continue
                    if entities[match]["modified"] == False:  # Propagate the changes
                        for k in ["entity", "links", "modified"]:
                            entities[match][k] = deepcopy(entities[(s, e)][k])
                    elif entities[match]["entity"] != entities[(s, e)]["entity"]:
                        if match > (s, e):
                            for k in ["entity", "links", "modified"]:
                                entities[match][k] = deepcopy(entities[(s, e)][k])
                    elif set(entities[match]["links"]) != set(
                        entities[(s, e)]["links"]
                    ):  # Two entities with same text have different annotations. BAD !!!
                        merged_links = set(entities[match]["links"]) | set(entities[(s, e)]["links"])
                        entities[match]["links"] = deepcopy(list(merged_links))
                        entities[(s, e)]["links"] = deepcopy(list(merged_links))
                    entities[match]["proped"] = True
                    add_match = False
                else:
                    for span in entities:
                        if overlap(span, match):
                            if entities[span]["modified"] == True:
                                add_match = False
                                if entities[span]["entity"] != entities[(s, e)]["entity"]:
                                    break
                                elif set(entities[span]["links"]) != set(entities[(s, e)]["links"]):
                                    diff_links = set(entities[(s, e)]["links"]) ^ set(entities[span]["links"])
                                    canon_name = set(["Canonical_Name"])
                                    if (
                                        diff_links != canon_name
                                        and set(entities[(s, e)]["links"]) != canon_name
                                        and set(entities[span]["links"]) != canon_name
                                    ):
                                        break
                                    else:
                                        merged_links = set(entities[(s, e)]["links"]) | set(entities[span]["links"])
                                        entities[(s, e)]["links"] = deepcopy(list(merged_links))
                                        entities[span]["links"] = deepcopy(list(merged_links))
                                        break
                                break
                        else:
                            add_match = True

                if match in new_entities:
                    if new_entities[match]["entity"] != entities[(s, e)]["entity"]:
                        breakpoint()
                    elif set(new_entities[match]["links"]) != set(
                        entities[(s, e)]["links"]
                    ):  # Two entities with same text have different annotations. BAD !!!
                        diff_links = set(new_entities[match]["links"]) & set(entities[(s, e)]["links"])
                        if (
                            len(diff_links) == 0
                            and len(set(new_entities[match]["links"])) > 0
                            and len(set(entities[(s, e)]["links"])) > 0
                        ):
                            breakpoint()
                        else:
                            merged_links = set(new_entities[match]["links"] + entities[(s, e)]["links"])
                            entities[(s, e)]["links"] = deepcopy(list(merged_links))
                            new_entities[match]["links"] = deepcopy(list(merged_links))
                    else:
                        add_match = False

                if add_match:
                    new_entities[match] = {k: deepcopy(entities[(s, e)][k]) for k in ["entity", "links", "modified"]}
                    new_entities[match]["token_start"] = match[0]
                    new_entities[match]["token_end"] = match[1]

    for match in list(new_entities.keys()):
        for span in list(entities.keys()):
            if overlap(match, span):
                assert entities[span]["modified"] == False or entities[span]["proped"], breakpoint()
                if entities[span].get("proped", False):
                    if match in new_entities :
                        del new_entities[match]
                elif not entities[span]["modified"]:
                    del entities[span]

    new_entities = sorted(list(new_entities.items()), key=lambda x: x[0][1])
    disjoint_new_entities = []
    for e in new_entities:
        if len(disjoint_new_entities) == 0:
            disjoint_new_entities.append(e)
        else:
            if e[0][0] >= disjoint_new_entities[-1][0][1]:
                disjoint_new_entities.append(e)

    assert not any(e[0] != f[0] and overlap(e[0], f[0]) for e in disjoint_new_entities for f in disjoint_new_entities)

    disjoint_new_entities = dict(disjoint_new_entities)
    assert not any(overlap(e, f) for e in disjoint_new_entities for f in entities), breakpoint()

    entities.update(disjoint_new_entities)
    assert not any(e != f and overlap(e, f) for e in entities for f in entities), breakpoint()
    assert all(v["token_start"] == s and v["token_end"] == e for (s, e), v in entities.items()), breakpoint()

    data_dict["entities"] = [x for x in entities.values()]

import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--annotator')

if __name__ == "__main__":
    args = parser.parse_args()

    annotations_dict = combine_brat_to_original_data(
        "data/pwc_s2_cleaned_text_v2.jsonl",
        "data/pwc_s2_cleaned_text_v2_sentences.jsonl",
        "outputs/pwc_s2_cleaned_text_v2_sentences_predictions.jsonl.clean",
        "/home/sarthakj/brat/brat/data/result_extraction/outputs/second_phase_annotations_" + args.annotator + "/",
        "/home/sarthakj/brat/brat/data/result_extraction/outputs/second_phase_annotations_original/",
    )

    annotations_to_jsonl(list(annotations_dict.values()), "model_data/all_data_" + args.annotator + ".jsonl")

    data = [json.loads(line) for line in open("model_data/all_data_" + args.annotator + ".jsonl")]
    for d in tqdm(data):
        names = [v for rel in d['relations'] for k, v in rel.items() if k != 'score']
        names += [n for m, subm in d['method_subrelations'].items() for idx, n in subm]
        names = set(names)
        propagate_annotations(d)

        coreference = {n: [] for n in names}
        ner = []
        for e in d['entities'] :
            e['links'] = set(e['links'])
            e['canon'] = 'Canonical_Name' in e['links']
            if e['canon'] :
                e['links'].remove('Canonical_Name')
            if 'proped' in e :
                del e['proped']
            del e['modified']
            e['links'] = e['links'] & names

            for l in e['links'] :
                coreference[l].append([e['token_start'], e['token_end']])

            ner.append((e['token_start'], e['token_end'], e['entity']))

        del d['entities']
        d['n_ary_relations'] = d['relations']
        del d['relations']
        d['coref'] = coreference
        d['ner'] = ner

        assert d['sentences'][-1][-1] == len(d['words']), breakpoint()
        assert d['sections'][-1][-1] == len(d['words']), breakpoint()

    annotations_to_jsonl(data, "model_data/all_data_" + args.annotator + "_propagated.jsonl")
