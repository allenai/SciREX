import json
import logging
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple, Union
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans
from allennlp.data.fields import (
    AdjacencyField,
    ListField,
    MetadataField,
    SequenceLabelField,
    SpanField,
    TextField,
    MultiLabelField,
)
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token

from dygie.data.dataset_readers.read_pwc_dataset import is_x_in_y, Relation, used_entities
from dygie.data.dataset_readers.paragraph_utils import *


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("pwc_json")
class PwCJsonReader(DatasetReader):
    """
    Reads a single JSON-formatted file. This is the same file format as used in the
    scierc, but is preprocessed
    """

    def __init__(
        self,
        max_span_width: int,
        token_indexers: Dict[str, TokenIndexer] = None,
        context_width: int = 1,
        max_paragraph_length=318,
        merge_paragraph_length=75,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy)
        assert (context_width % 2 == 1) and (context_width > 0)
        self.k = int((context_width - 1) / 2)
        self._max_span_width = max_span_width
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._max_paragraph_length = max_paragraph_length
        self._merge_paragraph_length = merge_paragraph_length

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as f:
            # Loop over the documents.
            for _, line in enumerate(f):
                js = json.loads(line)
                doc_key = js["doc_id"]

                paragraphs: List[(int, int)] = js["paragraphs"]
                words: List[str] = js["words"]
                entities: List[(int, int, str)] = js["ner"]
                corefs: Dict[str, List[(int, int)]] = js["coref"]
                n_ary_relations = js["n_ary_relations"]

                corefs_indexed = {}

                map_coref_keys = {k: i for i, k in enumerate(sorted(list(corefs.keys())))}
                for key in corefs:
                    for span in corefs[key]:
                        if tuple(span) not in corefs_indexed:
                            corefs_indexed[tuple(span)] = []
                        corefs_indexed[tuple(span)].append(map_coref_keys[key])

                corefs_indexed = {tuple(span): sorted(v) for span, v in corefs_indexed.items()}

                relations_indexed = {}
                for i, rel in enumerate(n_ary_relations):
                    rel = Relation(*rel)._asdict()
                    relations_indexed[i] = []
                    for entity in used_entities:
                        relations_indexed[i].append(map_coref_keys[rel[entity]])

                    relations_indexed[i] = sorted(relations_indexed[i])

                broken_paragraphs = move_boundaries(
                    break_paragraphs(
                        collapse_paragraphs(
                            paragraphs, min_len=self._merge_paragraph_length, max_len=self._max_paragraph_length
                        ),
                        max_len=self._max_paragraph_length,
                    ),
                    entities,
                )

                for p, q in zip(broken_paragraphs[:-1], broken_paragraphs[1:]):
                    if p[1] != q[0]:
                        breakpoint()

                paragraphs = broken_paragraphs
                entities_grouped = [{} for _ in range(len(paragraphs))]
                corefs_grouped = [{} for _ in range(len(paragraphs))]

                for e in entities:
                    done = False
                    for para_id, p in enumerate(paragraphs):
                        if is_x_in_y(e, p):
                            entities_grouped[para_id][(e[0] - p[0], e[1] - p[0])] = e[2]
                            corefs_grouped[para_id][(e[0] - p[0], e[1] - p[0])] = corefs_indexed.get((e[0], e[1]), [])
                            done = True

                    assert done

                zipped = zip(paragraphs, entities_grouped, corefs_grouped)

                # Loop over the sentences.
                for paragraph_num, ((start_ix, end_ix), ner_dict, coref_dict) in enumerate(zipped):
                    relation_dict = {}
                    paragraph = words[start_ix:end_ix]
                    if len(paragraph) == 0:
                        continue
                    instance = self.text_to_instance(
                        paragraph,
                        ner_dict,
                        relation_dict,
                        coref_dict,
                        doc_key,
                        paragraph_num,
                        start_ix,
                        end_ix,
                        map_coref_keys,
                        relations_indexed,
                    )
                    yield instance

    @overrides
    def text_to_instance(
        self,
        sentence: List[str],
        ner_dict: Dict[Tuple[int, int], str],
        relation_dict,
        cluster_dict,
        doc_key: str,
        sentence_num: int,
        start_ix: int,
        end_ix: int,
        map_coref_keys: Dict[str, int],
        relations_indexed: Dict[int, List[int]],
    ):
        """
        TODO(dwadden) document me.
        """

        sentence = [self._normalize_word(word) for word in sentence]

        text_field = TextField([Token(word) for word in sentence], self._token_indexers)
        assert len(map_coref_keys) > 0
        # Put together the metadata.
        metadata = dict(
            sentence=sentence,
            ner_dict=ner_dict,
            relation_dict=relation_dict,
            cluster_dict=cluster_dict,
            doc_key=doc_key,
            start_ix=0,
            end_ix=len(sentence),
            sentence_num=sentence_num,
            start_pos_in_doc=start_ix,
            end_pos_in_doc=end_ix,
            map_coref_keys=map_coref_keys,
            relations_indexed=relations_indexed
        )
        metadata_field = MetadataField(metadata)

        # Generate fields for text spans, ner labels, coref labels.
        spans = []
        span_ner_labels = []
        span_entity_labels = []
        span_link_labels = []
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

            label = '' if label in ['Method_False', 'Material_False'] else label
            span_ner_labels.append(label)
            span_entity_labels.append(label.split('_')[0] if label != '' else '')
            span_link_labels.append('True' if 'True' in label else '')

            span_coref_labels.append(
                MultiLabelField(
                    coref_label, label_namespace="coref_labels", skip_indexing=True, num_labels=len(map_coref_keys)
                )
            )
            spans.append(SpanField(start, end, text_field))

        span_field = ListField(spans)
        ner_label_field = SequenceLabelField(span_ner_labels, span_field, label_namespace="ner_labels")
        ner_entity_field = SequenceLabelField(span_entity_labels, span_field, label_namespace="ner_entity_labels")
        ner_link_field = SequenceLabelField(span_link_labels, span_field, label_namespace="ner_link_labels")
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
            ner_entity_labels=ner_entity_field,
            ner_link_labels=ner_link_field,
            coref_labels=coref_label_field,
            relation_index=relation_index_field,
            metadata=metadata_field,
        )

        return Instance(fields)

    @staticmethod
    def _normalize_word(word):
        if word == "/." or word == "/?":
            return word[1:]
        else:
            return word

    def print_instance(self, instance):
        tokens = instance.fields["text"].tokens
        spans = instance.fields["spans"].field_list
        span_labels = instance.fields["ner_labels"].labels

        for s, l in zip(spans, span_labels):
            if l != "":
                start, end = s.span_start, s.span_end + 1
                print(tokens[start:end])

    def validate_instances(self, instance_list):
        instances_by_doc = defaultdict(list)
        for instance in instance_list:
            instances_by_doc[instance.fields["metadata"].metadata["doc_key"]].append(instance)

        def convert_list_to_structured(instance_list):
            instance_list = sorted(instance_list, key=lambda x: x.fields["metadata"].metadata["sentence_num"])
            tokens = [w for ins in instance_list for w in ins.fields["text"].tokens]
            instance_start = [instance.fields["metadata"].metadata["start_pos_in_doc"] for instance in instance_list]

            entities = [
                (span.span_start + s, span.span_end + s + 1, span_label)
                for ins, s in zip(instance_list, instance_start)
                for span, span_label in zip(ins.fields["spans"].field_list, ins.fields["ner_labels"].labels)
                if span_label != ""
            ]

            return tokens, entities

        for key in instances_by_doc:
            instances_by_doc[key] = convert_list_to_structured(instances_by_doc[key])

        return instances_by_doc

