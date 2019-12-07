import json
import logging
from itertools import combinations
from typing import Any, Dict, List, Tuple

from collections import OrderedDict

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans
from allennlp.data.fields import ListField, MetadataField, MultiLabelField, SequenceLabelField, SpanField, TextField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token
from overrides import overrides

from dygie.data.dataset_readers.paragraph_utils import *
# from dygie.data.dataset_readers.read_pwc_dataset import Relation, is_x_in_y, used_entities
from scripts.entity_utils import *
from dygie.data.dataset_readers.span_utils import *

map_label = lambda x, n: "_".join([x[i] for i in n]) if x != "" else ""
is_x_in_y = lambda x, y: x[0] >= y[0] and x[1] <= y[1]

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

does_overlap = lambda x, y: max(x[0], y[0]) < min(x[1], y[1])


def clean_json_dict(json_dict):
    # Get fields from JSON dict
    entities: List[Tuple[int, int, str]] = json_dict["ner"]
    # Convert Entities to dictionary  and split type into tuple ("Entity", Entity Type, Is Linkable)
    entities = sorted(entities, key=lambda x: (x[0], x[1]))
    entities: Dict[Tuple[int, int], str] = OrderedDict([((s, e), t) for s, e, t in entities])

    corefs_all: Dict[str, List[Tuple[int, int]]] = json_dict["coref"]
    n_ary_relations_all: List[Dict[str, str]] = [x for x in json_dict["n_ary_relations"]]

    cluster_to_type = {}
    for rel in n_ary_relations_all:
        for k, v in rel.items():
            cluster_to_type[v] = k

    if "method_subrelations" in json_dict:
        method_subrelations: Dict[str, List[tuple]] = {
            x: [y[1] for y in v if y[1] != x] for x, v in json_dict["method_subrelations"].items()
        }

        for method, method_sub in method_subrelations.items():
            for m in method_sub:
                cluster_to_type[m] = "Method"

        for method_name, method_subparts in method_subrelations.items():
            for part in method_subparts:
                corefs_all[method_name].extend(corefs_all[part])

        for method_subparts in method_subrelations.values():
            for part in method_subparts:
                if part in corefs_all:
                    del corefs_all[part]

    for cluster, spans in corefs_all.items():
        corefs_all[cluster] = sorted(list(set([tuple(x) for x in spans])))

    # Remove clusters with no entries
    corefs = {k: v for k, v in corefs_all.items() if len(v) > 0}

    for cluster, spans in corefs.items():
        for span in spans:
            assert span in entities, breakpoint()
            if cluster not in cluster_to_type:
                continue
            entities[span] = cluster_to_type[cluster]

    for e in entities:
        entities[e] = tuple(["Entity", entities[e]] + [str(any(e in v for v in corefs.values()))])

    # Keep only relations where all clusters are non empty
    n_ary_relations = [r for r in n_ary_relations_all]# if all([v in corefs for k, v in r.items() if k in used_entities])]

    json_dict["coref"] = corefs
    json_dict["n_ary_relations"] = n_ary_relations
    json_dict["ner"] = entities
    if 'coref_non_salient' in json_dict :
        for cluster, spans in json_dict['coref_non_salient'].items():
            json_dict['coref_non_salient'][cluster] = sorted(list(set([tuple(x) for x in spans])))

    for e in entities:
        in_sentences = [i for i, s in enumerate(json_dict["sentences"]) if is_x_in_y(e, s)]
        if len(in_sentences) > 1:
            breakpoint()
        if len(in_sentences) == 0:
            in_sentences = [i for i, s in enumerate(json_dict["sentences"]) if does_overlap(e, s)]
            assert sorted(in_sentences) == list(range(min(in_sentences), max(in_sentences) + 1)), breakpoint()
            # breakpoint()
            in_sentences = sorted(in_sentences)
            json_dict["sentences"][in_sentences[0]][1] = json_dict["sentences"][in_sentences[-1]][1]
            json_dict["sentences"] = [s for i, s in enumerate(json_dict["sentences"]) if i not in in_sentences[1:]]

    json_dict["sentences"] = group_sentences_to_sections(json_dict["sentences"], json_dict["sections"])

    return json_dict


def group_sentences_to_sections(sentences, sections):
    grouped_sentences = [[] for _ in range(len(sections))]
    for s in sentences:
        done = 0
        for i, sec in enumerate(sections):
            if is_x_in_y(s, sec):
                grouped_sentences[i].append(s)
                done += 1
        if done != 1:
            breakpoint()

    return grouped_sentences


def verify_json_dict(json_dict):
    sentences: List[List[Tuple[int, int]]] = json_dict["sentences"]
    sections: List[Tuple[int, int]] = json_dict["sections"]
    entities: Dict[Tuple[int, int], str] = json_dict["ner"]
    corefs: Dict[str, List[Tuple[int, int]]] = json_dict["coref"]

    assert all(sum(is_x_in_y(e, s) for s in sections) == 1 for e in entities), breakpoint()
    assert all(sum(is_x_in_y(e, ss) for s in sentences for ss in s) == 1 for e in entities), breakpoint()
    assert all(
        (sections[i][0] == sentences[i][0][0] and sections[i][-1] == sentences[i][-1][-1]) for i in range(len(sections))
    ), breakpoint()

    assert all(x in entities for k, v in corefs.items() for x in v), breakpoint()


class PwCJsonReader(DatasetReader):
    def __init__(
        self,
        document_filter_type: str,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_paragraph_length: int = 300,
        merge_paragraph_length: int = 75,
        filter_to_salient: bool = False,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers
        self._max_paragraph_length = max_paragraph_length
        self._merge_paragraph_length = merge_paragraph_length
        self._filter_type = document_filter_type
        self.prediction_mode = False
        self._filter_to_salient = filter_to_salient

    @overrides
    def _read(self, file_path: str):
        try:
            file_path, dataset_ids = file_path.split(":")
            dataset_ids = dataset_ids.split(",")

            file_path = cached_path(file_path)
            with open(file_path, "r") as f:
                datasets = {k: v for k, v in json.load(f).items() if k in dataset_ids}
        except ValueError as e:
            datasets = {0: file_path}

        for dataset_path in datasets.values():
            with open(dataset_path, "r") as g:
                for idx, line in enumerate(g):
                    json_dict = json.loads(line)
                    if self.prediction_mode:
                        if 'method_subrelations' in json_dict :
                            del json_dict["method_subrelations"]
                        json_dict["n_ary_relations"] = []
                    json_dict = clean_json_dict(json_dict)

                    if self._filter_to_salient:
                        entities = json_dict["ner"]
                        entities = {k: v for k, v in entities.items() if v[2] == "True"}
                        json_dict["ner"] = entities

                    json_dict = filter_dispatcher[self._filter_type](json_dict)

                    verify_json_dict(json_dict)

                    # Get fields from JSON dict
                    doc_key = json_dict["doc_id"]
                    sentences: List[List[Tuple[int, int]]] = json_dict["sentences"]
                    sections: List[Tuple[int, int]] = json_dict["sections"]
                    words: List[str] = json_dict["words"]
                    entities: Dict[Tuple[int, int], str] = json_dict["ner"]
                    corefs: Dict[str, List[Tuple[int, int]]] = json_dict["coref"]
                    n_ary_relations: List[Dict[str, str]] = json_dict["n_ary_relations"]
                    corefs_non_salient: Dict[str, List[Tuple[int, int]]] = {} #json_dict.get("coref_non_salient", {})

                    # Extract Document structure features
                    entities_to_features_map = self.extract_section_based_features(sentences, words, entities)

                    # Map cluster names to integer cluster ids
                    cluster_name_to_id: Dict[str, int] = {k: i for i, k in enumerate(sorted(list(corefs.keys())))}
                    max_salient_cluster = len(corefs)
                    non_salient_cluster_name_to_id = {
                        k: max_salient_cluster + i for i, k in enumerate(sorted(list(corefs_non_salient.keys())))
                    }

                    # Map Spans to list of clusters ids it belong to.
                    span_to_cluster_ids: Dict[Tuple[int, int], List[int]] = {}
                    for cluster_name in corefs:
                        for span in corefs[cluster_name]:
                            span_to_cluster_ids.setdefault(span, []).append(cluster_name_to_id[cluster_name])

                    span_to_cluster_ids = {span: sorted(v) for span, v in span_to_cluster_ids.items()}

                    # Map Spans to list of non salient clusters they belong to.
                    span_to_non_salient_cluster_ids: Dict[Tuple[int, int], List[int]] = {}
                    for cluster_name in corefs_non_salient:
                        for span in corefs_non_salient[cluster_name]:
                            span_to_non_salient_cluster_ids.setdefault(span, []).append(
                                non_salient_cluster_name_to_id[cluster_name]
                            )

                    span_to_non_salient_cluster_ids = {
                        span: sorted(v) for span, v in span_to_non_salient_cluster_ids.items()
                    }

                    salient_cluster_label = [1] * max_salient_cluster + [0] * len(non_salient_cluster_name_to_id)
                    assert sorted(list(cluster_name_to_id.values())) == list(range(max_salient_cluster)), breakpoint()
                    assert sorted(list(non_salient_cluster_name_to_id.values())) == list(
                        range(max_salient_cluster, max_salient_cluster + len(non_salient_cluster_name_to_id))
                    ), breakpoint()

                    # Map types to list of cluster ids that are of that type
                    type_to_cluster_ids: Dict[str, List[int]] = {k: [] for k in used_entities}

                    for cluster_name in corefs:
                        types = [entities[span][1] for span in corefs[cluster_name]]
                        if len(set(types)) > 1:
                            print(json_dict["doc_id"])
                        if len(set(types)) == 1:
                            type_to_cluster_ids[types[0]].append(cluster_name_to_id[cluster_name])

                    # Map relations to list of cluster ids in it.
                    relation_to_cluster_ids: Dict[str, List[int]] = {}
                    for rel_idx, rel in enumerate(n_ary_relations):
                        relation_to_cluster_ids[rel_idx] = []
                        for entity in used_entities:
                            relation_to_cluster_ids[rel_idx].append(cluster_name_to_id[rel[entity]])
                            type_to_cluster_ids[entity].append(cluster_name_to_id[rel[entity]])

                        relation_to_cluster_ids[rel_idx] = tuple(relation_to_cluster_ids[rel_idx])

                    for k in type_to_cluster_ids:
                        type_to_cluster_ids[k] = sorted(list(set(type_to_cluster_ids[k])))

                    # Move paragraph boundaries around to accomodate in BERT
                    sections, sentences_grouped, entities_grouped = self.resize_paragraphs_and_group(
                        sections, sentences, entities
                    )

                    document_metadata = {
                        "cluster_labels" : salient_cluster_label,
                        "salient_cluster_label" : salient_cluster_label,
                        "span_to_cluster_ids": span_to_cluster_ids,
                        "cluster_name_to_id": cluster_name_to_id,
                        "span_to_non_salient_cluster_ids": span_to_non_salient_cluster_ids,
                        "non_salient_cluster_name_to_id": non_salient_cluster_name_to_id,
                        "relation_to_cluster_ids": relation_to_cluster_ids,
                        "type_to_cluster_ids": type_to_cluster_ids,
                        "doc_key": doc_key,
                        "doc_length": len(words),
                        "entities_to_features_map": entities_to_features_map,
                    }

                    # Loop over the sections.
                    for paragraph_num, ((start_ix, end_ix), sentences, ner_dict) in enumerate(
                        zip(sections, sentences_grouped, entities_grouped)
                    ):
                        paragraph = words[start_ix:end_ix]
                        if len(paragraph) == 0:
                            breakpoint()

                        instance = self.text_to_instance(
                            sentence=paragraph,
                            ner_dict=ner_dict,
                            sentence_num=paragraph_num,
                            start_ix=start_ix,
                            end_ix=end_ix,
                            sentence_indices=sentences,
                            document_metadata=document_metadata,
                        )
                        yield instance

    def extract_section_based_features(self, sentences, words, entities):
        entities_to_features_map = {}
        sentence_features = [get_features_for_sections(sents, words) for sents in sentences]
        for e in entities:
            index = [
                (i, j) for i, sents in enumerate(sentences) for j, sspan in enumerate(sents) if is_x_in_y(e, sspan)
            ]
            assert len(index) == 1, breakpoint()

            i, j = index[0]
            entities_to_features_map[(e[0], e[1])] = sentence_features[i][j]

        return entities_to_features_map

    def extract_coref_features(self, entities, words, span_to_cluster_ids):
        coref_labels = np.zeros((len(entities), len(entities)))
        coref_mask = np.zeros((len(entities), len(entities)))

        coref_features = np.zeros((len(entities), len(entities), span_feature_size))

        entity_keys = sorted(list(entities.keys()))
        for i, j in combinations(range(len(entity_keys)), 2):
            e1, e2 = entity_keys[i], entity_keys[j]
            if entities[e1][1] != entities[e2][1] or i == j:
                continue

            w1, w2 = " ".join(words[e1[0] : e1[1]]), " ".join(words[e2[0] : e2[1]])
            c1, c2 = set(span_to_cluster_ids.get((e1[0], e1[1]), [])), set(span_to_cluster_ids.get((e2[0], e2[1]), []))

            if w1.lower() == w2.lower() or len(c1 & c2) > 0:
                coref_labels[i, j] = 1
                coref_mask[i, j] = 1
                features = span_pair_features(e1, e2, w1, w2, words)
                coref_features[i, j] = features
                continue
            if len(c1) == 0 and len(c2) == 0:
                continue
            if len(c1 & c2) == 0:
                coref_mask[i, j] = 1
                features = span_pair_features(e1, e2, w1, w2, words)
                coref_features[i, j] = features
        return coref_labels, coref_mask, coref_features

    def resize_paragraphs_and_group(self, paragraphs: List[Tuple[int, int]], sentences, entities):
        broken_paragraphs = move_boundaries(
            break_paragraphs(
                collapse_paragraphs(paragraphs, min_len=20, max_len=self._max_paragraph_length),
                max_len=self._max_paragraph_length,
            ),
            list(entities.keys()),
        )

        # try :
        #     broken_paragraphs = break_and_collapse_sections(sentences, min_len=20, max_len=self._max_paragraph_length)
        # except :
        #     breakpoint()

        for p, q in zip(broken_paragraphs[:-1], broken_paragraphs[1:]):
            if p[1] != q[0] or p[1] < p[0] or q[1] < q[0]:
                breakpoint()

        paragraphs = broken_paragraphs
        entities_grouped = [{} for _ in range(len(paragraphs))]
        sentences_grouped = [[] for _ in range(len(paragraphs))]

        # Group entities into paragraphs they belong to
        for e in entities:
            is_in_n_para = 0
            for para_id, p in enumerate(paragraphs):
                if is_x_in_y(e, p):
                    entities_grouped[para_id][(e[0], e[1])] = entities[e]
                    is_in_n_para += 1

            assert is_in_n_para == 1, breakpoint()

        # for sec in sentences :
        #     for s in sec :
        #         is_in_n_para = 0
        #         for para_id, p in enumerate(paragraphs) :
        #             if is_x_in_y(s, p) :
        #                 sentences_grouped[para_id].append(s)
        #                 is_in_n_para += 1

        #         assert is_in_n_para == 1, breakpoint()

        zipped = zip(paragraphs, sentences_grouped, entities_grouped)

        # Remove Empty Paragraphs
        paragraphs, sentences_grouped, entities_grouped = [], [], []
        for p, s, e in zipped:
            if p[1] - p[0] == 0:
                assert len(e) == 0, breakpoint()
                assert len(s) == 0, breakpoint()
                continue
            paragraphs.append(p)
            entities_grouped.append(e)
            sentences_grouped.append(s)

        return paragraphs, sentences_grouped, entities_grouped

    def text_to_instance(self, *args, **kwargs):
        raise NotImplementedError


@DatasetReader.register("pwc_json_crf")
class PwCTagJsonReader(PwCJsonReader):
    def text_to_instance(
        self,
        sentence: List[str],
        ner_dict: Dict[Tuple[int, int], str],
        sentence_num: int,
        start_ix: int,
        end_ix: int,
        sentence_indices: List[Tuple[int, int]],
        document_metadata: Dict[str, Any],
    ):
        text_field = TextField([Token(word) for word in sentence], self._token_indexers)
        # Put together the metadata.

        cluster_name_to_id = document_metadata["cluster_name_to_id"]
        relation_to_cluster_ids = document_metadata["relation_to_cluster_ids"]
        span_to_cluster_ids = document_metadata["span_to_cluster_ids"]
        span_to_non_salient_cluster_ids = document_metadata["span_to_non_salient_cluster_ids"]
        non_salient_cluster_name_to_id = document_metadata["non_salient_cluster_name_to_id"]
        doc_key = document_metadata["doc_key"]

        metadata = dict(
            sentence=sentence,
            ner_dict=ner_dict,
            doc_key=doc_key,
            sentence_num=sentence_num,
            start_pos_in_doc=start_ix,
            end_pos_in_doc=end_ix,
            sentence_indices=sentence_indices,
            document_metadata=document_metadata,
            num_spans=len(ner_dict),
        )

        metadata_field = MetadataField(metadata)
        ner_is_entity_labels = spans_to_bio_tags(
            [(k[0] - start_ix, k[1] - start_ix, "_".join([v[0]])) for k, v in ner_dict.items()], len(sentence)
        )
        ner_entity_labels = spans_to_bio_tags(
            [(k[0] - start_ix, k[1] - start_ix, "_".join([v[0], v[1]])) for k, v in ner_dict.items()], len(sentence)
        )

        ner_entity_field = SequenceLabelField(ner_entity_labels, text_field, label_namespace="ner_entity_labels")
        ner_is_entity_field = SequenceLabelField(
            ner_is_entity_labels, text_field, label_namespace="ner_is_entity_labels"
        )

        # Pull it  all together.
        fields = dict(
            text=text_field,
            ner_entity_labels=ner_entity_field,
            ner_is_entity_labels=ner_is_entity_field,
            metadata=metadata_field,
        )

        spans = []
        span_coref_labels = []
        span_non_salient_coref_labels = []
        span_link_labels = []
        span_entity_labels = []
        span_features = []

        entities_to_features_map = document_metadata["entities_to_features_map"]

        for (s, e), label in ner_dict.items():
            spans.append(SpanField(int(s - start_ix), int(e - start_ix - 1), text_field))
            span_coref_labels.append(
                MultiLabelField(
                    span_to_cluster_ids.get((s, e), []) + span_to_non_salient_cluster_ids.get((s, e), []),
                    label_namespace="coref_labels",
                    skip_indexing=True,
                    num_labels=len(cluster_name_to_id) + len(non_salient_cluster_name_to_id),
                )
            )
            span_link_labels.append(1 if label[-1] == "True" else 0)
            span_entity_labels.append(label[1])
            span_features.append(
                MultiLabelField(entities_to_features_map[(s, e)], label_namespace="section_feature_labels")
            )

        if len(spans) > 0:
            fields["spans"] = ListField(spans)
            fields["span_coref_labels"] = ListField(span_coref_labels)
            fields["span_link_labels"] = SequenceLabelField(
                span_link_labels, fields["spans"], label_namespace="span_link_labels"
            )
            fields["span_entity_labels"] = SequenceLabelField(
                span_entity_labels, fields["spans"], label_namespace="span_entity_labels"
            )
            fields["span_features"] = ListField(span_features)
        else:
            fields["spans"] = ListField([SpanField(-1, -1, text_field).empty_field()]).empty_field()
            fields["span_coref_labels"] = ListField(
                [
                    MultiLabelField(
                        [], label_namespace="coref_labels", skip_indexing=True, num_labels=len(cluster_name_to_id) + len(non_salient_cluster_name_to_id)
                    )
                ]
            ).empty_field()
            fields["span_link_labels"] = SequenceLabelField([0], fields["spans"], label_namespace="span_link_labels")
            fields["span_entity_labels"] = SequenceLabelField(
                ["Method"], fields["spans"], label_namespace="span_entity_labels"
            )
            fields["span_features"] = ListField([MultiLabelField([], label_namespace="section_feature_labels")])

        if len(relation_to_cluster_ids) > 0:
            fields["relation_to_cluster_ids"] = ListField(
                [
                    MultiLabelField(
                        v, label_namespace="coref_labels", skip_indexing=True, num_labels=len(cluster_name_to_id)
                    )
                    for k, v in relation_to_cluster_ids.items()
                ]
            )

        return Instance(fields)
