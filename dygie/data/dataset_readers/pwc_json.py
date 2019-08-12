import json
import logging
from typing import Dict, List, Tuple
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans
from allennlp.data.fields import ListField, MetadataField, SequenceLabelField, SpanField, TextField, MultiLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token

from dygie.data.dataset_readers.read_pwc_dataset import is_x_in_y, Relation, used_entities
from dygie.data.dataset_readers.paragraph_utils import *
from dygie.data.dataset_readers.span_utils import *

map_label = lambda x, n: "_".join([x[i] for i in n]) if x != "" else ""

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
        context_width:int = 3, #TODO (Remove this - Legacy Argument)
        token_indexers: Dict[str, TokenIndexer] = None,
        max_paragraph_length=318,
        merge_paragraph_length=75,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy)
        self._max_span_width = max_span_width
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._max_paragraph_length = max_paragraph_length
        self._merge_paragraph_length = merge_paragraph_length

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path, dataset_ids = file_path.split(':')
        dataset_ids = dataset_ids.split(',')

        file_path = cached_path(file_path)
        with open(file_path, "r") as f:
            datasets = {k:v for k, v in json.load(f).items() if k in dataset_ids}

        for dataset_id, dataset_path in datasets.items():
            with open(dataset_path, "r") as g:
                for _, line in enumerate(g):
                    js = json.loads(line)
                    doc_key = js["doc_id"]

                    paragraphs: List[(int, int)] = js["paragraphs"]
                    words: List[str] = js["words"]
                    entities: List[(int, int, str)] = js["ner"]
                    corefs: Dict[str, List[(int, int)]] = js["coref"]
                    n_ary_relations: List[Relation] = [Relation(*x) for x in js["n_ary_relations"]]

                    for e in entities:
                        if e[-1] == 'Material_False' : 
                            e[-1] = 'Material_True'
                        e[-1] = tuple(["Entity"] + e[-1].split("_"))

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
                        rel = rel._asdict()
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
                                corefs_grouped[para_id][(e[0] - p[0], e[1] - p[0])] = corefs_indexed.get(
                                    (e[0], e[1]), []
                                )
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
        ner_dict: Dict[Tuple[int, int], Tuple[str]],
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


@DatasetReader.register("pwc_json_crf")
class PwCTagJsonReader(PwCJsonReader):
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
        # assert len(map_coref_keys) > 0

        ner_tag_labels = spans_to_bio_tags([(k[0], k[1], "_".join(v)) for k, v in ner_dict.items()], len(sentence))
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
            relations_indexed=relations_indexed,
        )
        metadata_field = MetadataField(metadata)

        ner_label_field = SequenceLabelField(ner_tag_labels, text_field, label_namespace="ner_labels")

        labels = [x.split("-") for x in ner_tag_labels]
        labels = [(x[0], x[1].split("_")) if len(x) > 1 else x for x in labels]
        ner_entity_labels = [x[0] + (("-" + map_label(x[1], (0, 1))) if len(x) > 1 else "") for x in labels]
        ner_link_labels = [x[0] + (("-" + map_label(x[1], (0, 2))) if len(x) > 1 else "") for x in labels]
        ner_is_entity_labels = [x[0] + (("-" + map_label(x[1], (0,))) if len(x) > 1 else "") for x in labels]

        ner_entity_field = SequenceLabelField(ner_entity_labels, text_field, label_namespace="ner_entity_labels")
        ner_link_field = SequenceLabelField(ner_link_labels, text_field, label_namespace="ner_link_labels")
        ner_is_entity_field = SequenceLabelField(
            ner_is_entity_labels, text_field, label_namespace="ner_is_entity_labels"
        )

        # coref_field = ListField(
        #     generate_seq_field(
        #         cluster_dict,
        #         len(sentence),
        #         lambda x: MultiLabelField(
        #             x if x is not None else [], label_namespace="coref_labels", skip_indexing=True, num_labels=len(map_coref_keys)
        #         ),
        #     )
        # )

        # Pull it  all together.
        fields = dict(
            text=text_field,
            ner_labels=ner_label_field,
            ner_entity_labels=ner_entity_field,
            ner_link_labels=ner_link_field,
            ner_is_entity_labels=ner_is_entity_field,
            # coref_labels=metadata_field,
            # relation_index=metadata_field,
            metadata=metadata_field,
        )

        return Instance(fields)
