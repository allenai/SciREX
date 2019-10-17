import json
import logging
from itertools import combinations
from typing import Any, Dict, List, Tuple

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans
from allennlp.data.fields import (
    ListField,
    MetadataField,
    MultiLabelField,
    SequenceLabelField,
    SpanField,
    TextField,
)
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token
from overrides import overrides

from dygie.data.dataset_readers.paragraph_utils import *
from dygie.data.dataset_readers.read_pwc_dataset import Relation, is_x_in_y, used_entities
from dygie.data.dataset_readers.span_utils import *

map_label = lambda x, n: "_".join([x[i] for i in n]) if x != "" else ""

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class PwCJsonReader(DatasetReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_paragraph_length=318,
        merge_paragraph_length=75,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers
        self._max_paragraph_length = max_paragraph_length
        self._merge_paragraph_length = merge_paragraph_length

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
                for i, line in enumerate(g):
                    js = json.loads(line)
                    doc_key = js["doc_id"]

                    paragraphs: List[(int, int)] = js["sections"]
                    words: List[str] = js["words"]
                    entities: List[(int, int, str)] = js["ner"]
                    corefs_all: Dict[str, List[(int, int)]] = js["coref"]
                    n_ary_relations_all: List[Relation] = [Relation(*x)._asdict() for x in js["n_ary_relations"]]

                    # Remove clusters with no entries
                    corefs = {k: v for k, v in corefs_all.items() if len(v) > 0}
                    n_ary_relations = [
                        r for r in n_ary_relations_all if all([v in corefs for k, v in r.items() if k in used_entities])
                    ]

                    section_features, entities_to_features_map = self.extract_section_based_features(paragraphs, words, entities)

                    for e in entities:
                        if e[-1] == "Material_False":
                            e[-1] = "Material_True"
                        e[-1] = tuple(["Entity"] + e[-1].split("_"))

                    entities_dict = {tuple([v[0], v[1]]): v[2] for v in entities}

                    #Map cluster names to integer cluster ids
                    map_coref_keys = {k: i for i, k in enumerate(sorted(list(corefs.keys())))}

                    #Map Spans to list of clusters ids it belong to.
                    corefs_indexed = {}
                    for key in corefs:
                        for span in corefs[key]:
                            if tuple(span) not in corefs_indexed:
                                corefs_indexed[tuple(span)] = []
                            corefs_indexed[tuple(span)].append(map_coref_keys[key])

                    corefs_indexed = {tuple(span): sorted(v) for span, v in corefs_indexed.items()}

                    #Map types to list of cluster ids that are of that type
                    type_to_clusters_map = {k: [] for k in used_entities}

                    for key in corefs:
                        types = [entities_dict[tuple(span)][1] for span in corefs[key]]
                        assert len(set(types)) == 1, breakpoint()
                        type_to_clusters_map[types[0]].append(map_coref_keys[key])

                    #Map relations to list of cluster ids in it.
                    relations_indexed = {}
                    for rel_idx, rel in enumerate(n_ary_relations):
                        relations_indexed[rel_idx] = []
                        for entity in used_entities:
                            relations_indexed[rel_idx].append(map_coref_keys[rel[entity]])
                            type_to_clusters_map[entity].append(map_coref_keys[rel[entity]])

                        relations_indexed[rel_idx] = tuple(relations_indexed[rel_idx])

                    for k in type_to_clusters_map:
                        type_to_clusters_map[k] = list(set(type_to_clusters_map[k]))

                    coref_labels, coref_mask, coref_features = self.extract_coref_features(
                        entities, words, corefs_indexed
                    )

                    ### Move paragraph boundaries around to accomodate in BERT
                    paragraphs, entities_grouped, corefs_grouped = self.resize_paragraphs_and_group(
                        paragraphs, entities, corefs_indexed
                    )

                    zipped = zip(paragraphs, entities_grouped, corefs_grouped)

                    document_metadata = {
                        "map_coref_keys": map_coref_keys,
                        "relations_indexed": relations_indexed,
                        "type_to_clusters_map": type_to_clusters_map,
                        "doc_key": doc_key,
                        "doc_length": len(words),
                        "coref_labels": coref_labels,
                        "coref_mask": coref_mask,
                        "coref_features": coref_features,
                        "entities_to_features_map": entities_to_features_map,
                        "section_features": section_features
                    }
                    # Loop over the sentences.
                    for paragraph_num, ((start_ix, end_ix), ner_dict, coref_dict) in enumerate(zipped):
                        paragraph = words[start_ix:end_ix]
                        if len(paragraph) == 0:
                            breakpoint()
                        instance = self.text_to_instance(
                            paragraph, ner_dict, coref_dict, paragraph_num, start_ix, end_ix, document_metadata
                        )
                        yield instance

    def extract_section_based_features(self, paragraphs, words, entities):
        section_features = get_features_for_sections(paragraphs, words)
        assert len(section_features) == len(paragraphs)

        entities_to_features_map = {}
        for e in entities:
            entities_to_features_map[(e[0], e[1])] = [
                f for (start, end), f in zip(paragraphs, section_features) if is_x_in_y((e[0], e[1]), (start, end))
            ]

            assert len(entities_to_features_map[(e[0], e[1])]) == 1, breakpoint()

            entities_to_features_map[(e[0], e[1])] = entities_to_features_map[(e[0], e[1])][0]
        return section_features, entities_to_features_map

    def extract_coref_features(self, entities, words, corefs_indexed):
        coref_labels = np.zeros((len(entities), len(entities)))
        coref_mask = np.zeros((len(entities), len(entities)))

        coref_features = np.zeros((len(entities), len(entities), span_feature_size))

        for i, j in combinations(range(len(entities)), 2):
            e1, e2 = entities[i], entities[j]
            if e1[-1][1] != e2[-1][1] or i == j:
                continue

            w1, w2 = " ".join(words[e1[0] : e1[1]]), " ".join(words[e2[0] : e2[1]])
            c1, c2 = set(corefs_indexed.get((e1[0], e1[1]), [])), set(corefs_indexed.get((e2[0], e2[1]), []))

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

    def resize_paragraphs_and_group(self, paragraphs, entities, corefs_indexed):
        broken_paragraphs = move_boundaries(
            break_paragraphs(
                collapse_paragraphs(paragraphs, min_len=20, max_len=self._max_paragraph_length),
                max_len=self._max_paragraph_length,
            ),
            entities,
        )

        for p, q in zip(broken_paragraphs[:-1], broken_paragraphs[1:]):
            if p[1] != q[0] or p[1] < p[0] or q[1] < q[0]:
                breakpoint()

        paragraphs = broken_paragraphs
        entities_grouped = [{} for _ in range(len(paragraphs))]
        corefs_grouped = [{} for _ in range(len(paragraphs))]

        #Group entities into paragraphs they belong to
        for e in entities:
            done = False
            for para_id, p in enumerate(paragraphs):
                if is_x_in_y(e, p):
                    entities_grouped[para_id][(e[0], e[1])] = e[2]
                    corefs_grouped[para_id][(e[0], e[1])] = corefs_indexed.get((e[0], e[1]), [])
                    done = True

            assert done

        zipped = zip(paragraphs, entities_grouped, corefs_grouped)

        #Remove Empty Paragraphs
        paragraphs, entities_grouped, corefs_grouped = [], [], []
        for p, e, c in zipped:
            if p[1] - p[0] == 0:
                assert len(e) == 0 and len(c) == 0, breakpoint()
                continue
            paragraphs.append(p)
            entities_grouped.append(e)
            corefs_grouped.append(c)

        return paragraphs, entities_grouped, corefs_grouped

    def text_to_instance(self, *args, **kwargs):
        raise NotImplementedError


@DatasetReader.register("pwc_json_span")
class PwCSpanJsonReader(PwCJsonReader):
    def __init__(
        self,
        max_span_width: int = 8,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_paragraph_length=318,
        merge_paragraph_length=75,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy=lazy)
        self._max_span_width = max_span_width
        self._token_indexers = token_indexers
        self._max_paragraph_length = max_paragraph_length
        self._merge_paragraph_length = merge_paragraph_length

    @overrides
    def text_to_instance(
        self,
        sentence: List[str],
        ner_dict: Dict[Tuple[int, int], Tuple[str]],
        cluster_dict,
        doc_key: str,
        sentence_num: int,
        start_ix: int,
        end_ix: int,
        map_coref_keys: Dict[str, int],
        relations_indexed: Dict[int, List[int]],
    ):
        text_field = TextField([Token(word) for word in sentence], self._token_indexers)
        assert len(map_coref_keys) > 0
        # Put together the metadata.
        metadata = dict(
            sentence=sentence,
            ner_dict=ner_dict,
            cluster_dict=cluster_dict,
            doc_key=doc_key,
            start_ix=0,
            end_ix=len(sentence),
            sentence_num=sentence_num,
            start_pos_in_doc=start_ix,
            end_pos_in_doc=end_ix,
            map_coref_keys=map_coref_keys,
            relations_indexed=relations_indexed,
        )
        metadata_field = MetadataField(metadata)

        # Generate fields for text spans, ner labels, coref labels.
        spans = []
        span_ner_labels = []
        span_coref_labels = []
        for start, end in enumerate_spans(sentence, max_span_width=self._max_span_width):
            span_ix = (start, end + 1)
            label = ner_dict.get(span_ix, "")
            coref_label = cluster_dict.get(span_ix, [])
            if end - start + 1 == self._max_span_width and label == "":
                for e in ner_dict:
                    if is_x_in_y(span_ix, e):
                        label = ner_dict[e]
                        coref_label = cluster_dict[e]

            span_ner_labels.append(map_label(label, (0, 1, 2)))

            span_coref_labels.append(
                MultiLabelField(
                    coref_label, label_namespace="coref_labels", skip_indexing=True, num_labels=len(map_coref_keys)
                )
            )
            spans.append(SpanField(start, end, text_field))

        span_field = ListField(spans)
        ner_label_field = SequenceLabelField(span_ner_labels, span_field, label_namespace="ner_labels")
        coref_label_field = ListField(span_coref_labels)

        relation_index_field = ListField(
            [
                MultiLabelField(v, label_namespace="coref_labels", skip_indexing=True, num_labels=len(map_coref_keys))
                for k, v in relations_indexed.items()
            ]
        )

        # Pull it  all together.
        fields = dict(
            text=text_field,
            spans=span_field,
            ner_labels=ner_label_field,
            coref_labels=coref_label_field,
            relation_index=relation_index_field,
            metadata=metadata_field,
        )

        return Instance(fields)


@DatasetReader.register("pwc_json_crf")
class PwCTagJsonReader(PwCJsonReader):
    def text_to_instance(
        self,
        sentence: List[str],
        ner_dict: Dict[Tuple[int, int], str],
        cluster_dict: Dict[Tuple[int, int], List[int]],
        sentence_num: int,
        start_ix: int,
        end_ix: int,
        document_metadata: Dict[str, Any],
    ):
        text_field = TextField([Token(word) for word in sentence], self._token_indexers)
        # Put together the metadata.

        map_coref_keys = document_metadata["map_coref_keys"]
        relations_indexed = document_metadata["relations_indexed"]
        doc_key = document_metadata["doc_key"]

        metadata = dict(
            sentence=sentence,
            ner_dict=ner_dict,
            cluster_dict=cluster_dict,
            doc_key=doc_key,
            sentence_num=sentence_num,
            start_pos_in_doc=start_ix,
            end_pos_in_doc=end_ix,
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
        span_link_labels = []
        span_entity_labels = []
        span_features = []

        entities_to_features_map = document_metadata["entities_to_features_map"]

        for (s, e), label in ner_dict.items():
            spans.append(SpanField(int(s - start_ix), int(e - start_ix - 1), text_field))
            span_coref_labels.append(
                MultiLabelField(
                    cluster_dict[(s, e)],
                    label_namespace="coref_labels",
                    skip_indexing=True,
                    num_labels=len(map_coref_keys),
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
            fields['span_features'] = ListField(span_features)
        else:
            fields["spans"] = ListField([SpanField(-1, -1, text_field).empty_field()]).empty_field()
            fields["span_coref_labels"] = ListField(
                [
                    MultiLabelField(
                        [], label_namespace="coref_labels", skip_indexing=True, num_labels=len(map_coref_keys)
                    )
                ]
            ).empty_field()
            fields["span_link_labels"] = SequenceLabelField([0], fields["spans"], label_namespace="span_link_labels")
            fields["span_entity_labels"] = SequenceLabelField(
                ["Method"], fields["spans"], label_namespace="span_entity_labels"
            )
            fields['span_features'] = ListField([MultiLabelField([], label_namespace="section_feature_labels")])

        if len(relations_indexed) > 0:
            fields["relation_index"] = ListField(
                [
                    MultiLabelField(
                        v, label_namespace="coref_labels", skip_indexing=True, num_labels=len(map_coref_keys)
                    )
                    for k, v in relations_indexed.items()
                ]
            )

        return Instance(fields)
