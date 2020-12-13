import json
from collections import OrderedDict
from typing import Any, Dict, List, Set, Tuple

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (
    ListField,
    MetadataField,
    SequenceLabelField,
    SpanField,
    TextField,
)
from scirex.data.dataset_readers.multi_label_field import MultiLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token
from overrides import overrides

from scirex.data.utils.paragraph_alignment import *
from scirex.data.utils.section_feature_extraction import extract_sentence_features
from scirex.data.utils.span_utils import does_overlap, is_x_in_y, spans_to_bio_tags
from scirex_utilities.entity_utils import used_entities

from scipy.stats import mode

Span = Tuple[int, int]  # eg. (0, 10)
ClusterName = str
BaseEntityType = str  # eg. Method
EntityType = Tuple[str, str]  # eg. (Method, True)


def clean_json_dict(json_dict):
    # Get fields from JSON dict
    entities: List[Tuple[int, int, BaseEntityType]] = json_dict.get("ner", [])
    # Convert Entities to dictionary {(s, e) -> type}
    entities = sorted(entities, key=lambda x: (x[0], x[1]))
    entities: Dict[Span, BaseEntityType] = OrderedDict([((s, e), t) for s, e, t in entities])

    clusters_dict: Dict[ClusterName, List[Span]] = {
        cluster_name: sorted(list(set([tuple(x) for x in spans])))
        for cluster_name, spans in json_dict.get('coref', dict()).items()
    }

    n_ary_relations: List[Dict[BaseEntityType, ClusterName]] = [x for x in json_dict.get("n_ary_relations", list())]
    existing_entities = set([v for relation in n_ary_relations for k, v in relation.items()])

    cluster_to_type: Dict[ClusterName, BaseEntityType] = {}
    for rel in n_ary_relations:
        for k, v in rel.items():
            cluster_to_type[v] = k

    # Under current model, we do not use method subrelations as separate component
    # Therefore, we add each submethod as a separate entity
    if "method_subrelations" in json_dict:
        # Map each method to set containing (all submethod names and the method name itself) .
        method_subrelations: Dict[ClusterName, Set[ClusterName]] = {
            k: set([k] + [x[1] for x in v]) for k, v in json_dict["method_subrelations"].items()
        }

        # Add each submethod to cluster_to_type as Method
        for method_name, method_sub in method_subrelations.items():
            for m in method_sub:
                if m in clusters_dict and m != method_name and m not in existing_entities:
                    clusters_dict[method_name] += clusters_dict[m]
                    clusters_dict[method_name] = sorted(list(set(clusters_dict[method_name])))

                    del clusters_dict[m]

    for cluster, spans in clusters_dict.items():
        for span in spans:
            assert span in entities, breakpoint()
            if cluster not in cluster_to_type:
                continue
            entities[span] = cluster_to_type[cluster]

    for e in entities:
        entities[e]: EntityType = (entities[e], str(any(e in v for v in clusters_dict.values())))

    json_dict["ner"]: Dict[Span, BaseEntityType] = entities
    json_dict["coref"]: Dict[ClusterName, List[Span]] = clusters_dict

    for e in entities:
        in_sentences = [
            i for i, s in enumerate(json_dict["sentences"]) if is_x_in_y(e, s)
        ]  # Check entity lie in one sentence
        if len(in_sentences) > 1:
            breakpoint()
        if len(in_sentences) == 0:
            in_sentences = [i for i, s in enumerate(json_dict["sentences"]) if does_overlap(e, s)]
            assert sorted(in_sentences) == list(range(min(in_sentences), max(in_sentences) + 1)), breakpoint()
            # breakpoint()
            in_sentences = sorted(in_sentences)
            json_dict["sentences"][in_sentences[0]][1] = json_dict["sentences"][in_sentences[-1]][1]
            json_dict["sentences"] = [
                s for i, s in enumerate(json_dict["sentences"]) if i not in in_sentences[1:]
            ]

    json_dict["sentences"]: List[List[Span]] = group_sentences_to_sections(
        json_dict["sentences"], json_dict["sections"]
    )

    return json_dict


def verify_json_dict(json_dict):
    sentences: List[List[Span]] = json_dict["sentences"]
    sections: List[Span] = json_dict["sections"]
    entities: Dict[Span, str] = json_dict["ner"]
    corefs: Dict[str, List[Span]] = json_dict["coref"]

    assert all(sum(is_x_in_y(e, s) for s in sections) == 1 for e in entities), breakpoint()
    assert all(sum(is_x_in_y(e, ss) for s in sentences for ss in s) == 1 for e in entities), breakpoint()
    assert all(
        (sections[i][0] == sentences[i][0][0] and sections[i][-1] == sentences[i][-1][-1])
        for i in range(len(sections))
    ), breakpoint()

    assert all(x in entities for k, v in corefs.items() for x in v), breakpoint()


@DatasetReader.register("scirex_full_reader")
class ScirexFullReader(DatasetReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_paragraph_length: int = 300,
        lazy: bool = False,
        to_scirex_converter: bool = False,
    ) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers
        self._max_paragraph_length = max_paragraph_length
        self.prediction_mode = False

        ## Hack so I can reuse same reader to convert scirex
        ## format to scierc format
        self.to_scierc_converter = to_scirex_converter

    @overrides
    def _read(self, file_path: str):
        with open(file_path, "r") as g:
            for _, line in enumerate(g):
                json_dict = json.loads(line)
                if self.prediction_mode:
                    if "method_subrelations" in json_dict:
                        del json_dict["method_subrelations"]
                    json_dict["n_ary_relations"] = []
                json_dict = clean_json_dict(json_dict)

                verify_json_dict(json_dict)

                # Get fields from JSON dict
                doc_id = json_dict["doc_id"]
                sections: List[Span] = json_dict["sections"]
                sentences: List[List[Span]] = json_dict["sentences"]
                words: List[str] = json_dict["words"]
                entities: Dict[Span, EntityType] = json_dict["ner"]
                corefs: Dict[ClusterName, List[Span]] = json_dict["coref"]
                n_ary_relations: List[Dict[BaseEntityType, ClusterName]] = json_dict["n_ary_relations"]

                # Extract Document structure features
                entities_to_features_map: Dict[Span, List[str]] = extract_sentence_features(
                    sentences, words, entities
                )

                # Map cluster names to integer cluster ids
                cluster_name_to_id: Dict[ClusterName, int] = {
                    k: i for i, k in enumerate(sorted(list(corefs.keys())))
                }
                max_salient_cluster = len(corefs)

                # Map Spans to list of clusters ids it belong to.
                span_to_cluster_ids: Dict[Span, List[int]] = {}
                for cluster_name in corefs:
                    for span in corefs[cluster_name]:
                        span_to_cluster_ids.setdefault(span, []).append(cluster_name_to_id[cluster_name])

                span_to_cluster_ids = {span: sorted(v) for span, v in span_to_cluster_ids.items()}

                assert sorted(list(cluster_name_to_id.values())) == list(
                    range(max_salient_cluster)
                ), breakpoint()

                # Map types to list of cluster ids that are of that type
                type_to_cluster_ids: Dict[BaseEntityType, List[int]] = {k: [] for k in used_entities}

                for cluster_name in corefs:
                    types = [entities[span][0] for span in corefs[cluster_name]]
                    if len(set(types)) > 0:
                        try :
                            type_to_cluster_ids[mode(types)[0][0]].append(cluster_name_to_id[cluster_name])
                        except :
                            # SciERC gives trouble here. Not relevant .
                            continue

                # Map relations to list of cluster ids in it.
                relation_to_cluster_ids: Dict[int, List[int]] = {}
                for rel_idx, rel in enumerate(n_ary_relations):
                    relation_to_cluster_ids[rel_idx] = []
                    for entity in used_entities:
                        relation_to_cluster_ids[rel_idx].append(cluster_name_to_id[rel[entity]])
                        type_to_cluster_ids[entity].append(cluster_name_to_id[rel[entity]])

                    relation_to_cluster_ids[rel_idx] = tuple(relation_to_cluster_ids[rel_idx])

                for k in type_to_cluster_ids:
                    type_to_cluster_ids[k] = sorted(list(set(type_to_cluster_ids[k])))

                # Move paragraph boundaries around to accomodate in BERT
                sections, sentences_grouped, entities_grouped = self.resize_sections_and_group(
                    sections, sentences, entities
                )

                document_metadata = {
                    "cluster_name_to_id": cluster_name_to_id,
                    "span_to_cluster_ids": span_to_cluster_ids,
                    "relation_to_cluster_ids": relation_to_cluster_ids,
                    "type_to_cluster_ids": type_to_cluster_ids,
                    "doc_id": doc_id,
                    "doc_length": len(words),
                    "entities_to_features_map": entities_to_features_map,
                }

                # Loop over the sections.
                for (paragraph_num, ((start_ix, end_ix), sentences, ner_dict)) in enumerate(
                    zip(sections, sentences_grouped, entities_grouped)
                ):
                    paragraph = words[start_ix:end_ix]
                    if len(paragraph) == 0:
                        breakpoint()

                    instance = self.text_to_instance(
                        paragraph_num=paragraph_num,
                        paragraph=paragraph,
                        ner_dict=ner_dict,
                        start_ix=start_ix,
                        end_ix=end_ix,
                        sentence_indices=sentences,
                        document_metadata=document_metadata,
                    )
                    yield instance

    def resize_sections_and_group(
        self, sections: List[Span], sentences: List[List[Span]], entities: Dict[Span, EntityType]
    ):
        broken_sections = move_boundaries(
            break_paragraphs(
                collapse_paragraphs(sections, min_len=20, max_len=self._max_paragraph_length),
                max_len=self._max_paragraph_length,
            ),
            list(entities.keys()),
        )

        for p, q in zip(broken_sections[:-1], broken_sections[1:]):
            if p[1] != q[0] or p[1] < p[0] or q[1] < q[0]:
                breakpoint()

        sections = broken_sections
        entities_grouped = [{} for _ in range(len(sections))]
        sentences_grouped = [[] for _ in range(len(sections))]

        # Bert is PITA. Group entities into sections they belong to.
        for e in entities:
            is_in_n_para = 0
            for para_id, p in enumerate(sections):
                if is_x_in_y(e, p):
                    entities_grouped[para_id][(e[0], e[1])] = entities[e]
                    is_in_n_para += 1

            assert is_in_n_para == 1, breakpoint()

        ## Bert is serious PITA. Need to align sentences with sections also.
        sentences = [sent for section in sentences for sent in section]
        assert all([sentences[i + 1][0] == sentences[i][1] for i in range(len(sentences) - 1)]), breakpoint()
        assert sentences[-1][1] == sections[-1][1], breakpoint()
        sentence_indices = sorted(list(set([0] + [s[1] for s in sentences] + [s[1] for s in sections])))
        sentences = list(zip(sentence_indices[:-1], sentence_indices[1:]))
        for e in sentences:
            is_in_n_para = 0
            for para_id, p in enumerate(sections):
                if is_x_in_y(e, p):
                    sentences_grouped[para_id].append(e)
                    is_in_n_para += 1

            assert is_in_n_para == 1, breakpoint()

        zipped = zip(sections, sentences_grouped, entities_grouped)

        # Remove Empty sections
        sections, sentences_grouped, entities_grouped = [], [], []
        for p, s, e in zipped:
            if p[1] - p[0] == 0:
                assert len(e) == 0, breakpoint()
                assert len(s) == 0, breakpoint()
                continue
            sections.append(p)
            entities_grouped.append(e)
            sentences_grouped.append(s)

        return sections, sentences_grouped, entities_grouped

    def text_to_instance(
        self,
        paragraph_num: int,
        paragraph: List[str],
        ner_dict: Dict[Span, str],
        start_ix: int,
        end_ix: int,
        sentence_indices: List[Span],
        document_metadata: Dict[str, Any],
    ):

        if self.to_scierc_converter:
            return dict(
                paragraph_num=paragraph_num,
                paragraph=paragraph,
                ner_dict=ner_dict,
                start_ix=start_ix,
                end_ix=end_ix,
                sentence_indices=sentence_indices,
                document_metadata=document_metadata,
            )

        text_field = TextField([Token(word) for word in paragraph], self._token_indexers)

        metadata_field = MetadataField(
            dict(
                doc_id=document_metadata["doc_id"],
                paragraph_num=paragraph_num,
                paragraph=paragraph,
                start_pos_in_doc=start_ix,
                end_pos_in_doc=end_ix,
                ner_dict=ner_dict,
                sentence_indices=sentence_indices,
                document_metadata=document_metadata,
                num_spans=len(ner_dict),
            )
        )

        ner_type_labels = spans_to_bio_tags(
            [(k[0] - start_ix, k[1] - start_ix, v[0]) for k, v in ner_dict.items()], len(paragraph)
        )

        ner_entity_field = SequenceLabelField(ner_type_labels, text_field, label_namespace="ner_type_labels")

        # Pull it  all together.
        fields = dict(text=text_field, ner_type_labels=ner_entity_field, metadata=metadata_field)

        spans = []
        span_cluster_labels = []
        span_saliency_labels = []
        span_type_labels = []
        span_features = []

        entities_to_features_map = document_metadata["entities_to_features_map"]
        cluster_name_to_id = document_metadata["cluster_name_to_id"]
        relation_to_cluster_ids = document_metadata["relation_to_cluster_ids"]
        span_to_cluster_ids = document_metadata["span_to_cluster_ids"]

        for (s, e), label in ner_dict.items():
            spans.append(SpanField(int(s - start_ix), int(e - start_ix - 1), text_field))
            span_cluster_labels.append(
                MultiLabelField(
                    span_to_cluster_ids.get((s, e), []),
                    label_namespace="cluster_labels",
                    skip_indexing=True,
                    num_labels=len(cluster_name_to_id),
                )
            )
            span_saliency_labels.append(1 if label[-1] == "True" else 0)
            span_type_labels.append(label[0])
            span_features.append(
                MultiLabelField(entities_to_features_map[(s, e)], label_namespace="section_feature_labels", num_labels=5)
            )

        if len(spans) > 0:
            fields["spans"] = ListField(spans)
            fields["span_cluster_labels"] = ListField(span_cluster_labels)
            fields["span_saliency_labels"] = SequenceLabelField(
                span_saliency_labels, fields["spans"], label_namespace="span_saliency_labels"
            )
            fields["span_type_labels"] = SequenceLabelField(
                span_type_labels, fields["spans"], label_namespace="span_type_labels"
            )
            fields["span_features"] = ListField(span_features)
        else:  # Some paragraphs may not have anything !
            fields["spans"] = ListField([SpanField(-1, -1, text_field).empty_field()]).empty_field()
            fields["span_cluster_labels"] = ListField(
                [
                    MultiLabelField(
                        [],
                        label_namespace="cluster_labels",
                        skip_indexing=True,
                        num_labels=len(cluster_name_to_id),
                    )
                ]
            ) #.empty_field()
            fields["span_saliency_labels"] = SequenceLabelField(
                [0], fields["spans"], label_namespace="span_saliency_labels"
            )
            fields["span_type_labels"] = SequenceLabelField(
                ["Method"], fields["spans"], label_namespace="span_type_labels"
            )
            fields["span_features"] = ListField(
                [MultiLabelField([], label_namespace="section_feature_labels", num_labels=5)]
            )

        if len(relation_to_cluster_ids) > 0:
            fields["relation_to_cluster_ids"] = ListField(
                [
                    MultiLabelField(
                        v,
                        label_namespace="cluster_labels",
                        skip_indexing=True,
                        num_labels=len(cluster_name_to_id),
                    )
                    for k, v in relation_to_cluster_ids.items()
                ]
            )

        return Instance(fields)
