import json
import logging
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple, Union
import numpy as np
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans
from allennlp.data.fields import AdjacencyField, ListField, MetadataField, SequenceLabelField, SpanField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token

from dygie.data.dataset_readers.read_pwc_dataset import is_x_in_y

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
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy)
        assert (context_width % 2 == 1) and (context_width > 0)
        self.k = int((context_width - 1) / 2)
        self._max_span_width = max_span_width
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._max_paragraph_length = 400

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as f:
            # Loop over the documents.
            for i, line in enumerate(f):
                js = json.loads(line)
                doc_key = js["doc_id"]

                # TODO(Sarthak) : Rename sentences to paragraphs
                paragraphs: List[(int, int)] = js["paragraphs"]
                words: List[str] = js["words"]
                entities: List[(int, int, str)] = js["ner"]

                # entity_length = [e[1] - e[0] for e in entities]
                # print(max(entity_length))
                # max_e = entities[np.argmax(entity_length)]
                # print(words[max_e[0] : max_e[1]], max_e[2])

                entities_grouped = [{} for _ in range(len(paragraphs))]
                for e in entities:
                    for para_id, p in enumerate(paragraphs):
                        if is_x_in_y(e, p):
                            entities_grouped[para_id][(e[0] - p[0], e[1] - p[0])] = e[2]

                broken_paragraphs = []
                for p, e in zip(paragraphs, entities_grouped):
                    plen = p[1] - p[0]
                    if plen < self._max_paragraph_length:
                        broken_paragraphs.append(p)
                    else:
                        protected_indices = sorted([i for x in e for i in range(x[0] + 1, x[1])])
                        splits = list(range(0, plen, self._max_paragraph_length))
                        for split_index, s in enumerate(splits):
                            if s in protected_indices:
                                while True:
                                    s = s + 1
                                    assert s < splits[split_index + 1] if split_index + 1 < len(splits) else True
                                    if s not in protected_indices:
                                        splits[split_index] = s
                                        break

                        assert len(set(splits) & set(protected_indices)) == 0, (protected_indices, splits)
                        new_para = []
                        if splits[0] != 0:
                            splits = [0] + splits
                        for start, end in zip(splits, splits[1:] + [plen]):
                            if p[0] + end > p[1]:
                                breakpoint()
                            new_para.append([p[0] + start, p[0] + end])

                        if len(new_para) == 0:
                            breakpoint()
                        broken_paragraphs += new_para

                for p, q in zip(broken_paragraphs[:-1], broken_paragraphs[1:]):
                    if p[1] != q[0]:
                        breakpoint()

                paragraphs = broken_paragraphs
                entities_grouped = [{} for _ in range(len(paragraphs))]
                for e in entities:
                    done = False
                    for para_id, p in enumerate(paragraphs):
                        if is_x_in_y(e, p):
                            entities_grouped[para_id][(e[0] - p[0], e[1] - p[0])] = e[2]
                            done = True

                    assert done

                zipped = zip(paragraphs, entities_grouped)

                # Loop over the sentences.
                for sentence_num, ((start_ix, end_ix), ner_dict) in enumerate(zipped):
                    relation_dict = {}
                    cluster_dict = {}
                    sentence = words[start_ix:end_ix]
                    instance = self.text_to_instance(
                        sentence, ner_dict, relation_dict, cluster_dict, doc_key, sentence_num, start_ix, end_ix
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
    ):
        """
        TODO(dwadden) document me.
        """

        sentence = [self._normalize_word(word) for word in sentence]

        text_field = TextField([Token(word) for word in sentence], self._token_indexers)

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
        )
        metadata_field = MetadataField(metadata)

        # Generate fields for text spans, ner labels, coref labels.
        spans = []
        span_ner_labels = []
        span_coref_labels = []
        for start, end in enumerate_spans(sentence, max_span_width=self._max_span_width):
            span_ix = (start, end + 1)
            label = ner_dict.get(span_ix, "")
            if end - start + 1 == self._max_span_width and label == "":
                for e in ner_dict:
                    if is_x_in_y(span_ix, e):
                        label = ner_dict[e]

            span_ner_labels.append(label)
            span_coref_labels.append(cluster_dict.get(span_ix, -1))
            spans.append(SpanField(start, end, text_field))

        span_field = ListField(spans)
        ner_label_field = SequenceLabelField(span_ner_labels, span_field, label_namespace="ner_labels")
        coref_label_field = SequenceLabelField(span_coref_labels, span_field, label_namespace="coref_labels")

        # Generate labels for relations. Only store non-null values.
        # n_spans = len(spans)
        # span_tuples = [(span.span_start, span.span_end) for span in spans]
        # candidate_indices = [(i, j) for i in range(n_spans) for j in range(n_spans)]

        relations = []
        relation_indices = []
        # for i, j in candidate_indices:
        #     span_pair = (span_tuples[i], span_tuples[j])
        #     relation_label = relation_dict.get(span_pair, "")
        #     if relation_label:
        #         relation_indices.append((i, j))
        #         relations.append(relation_label)

        relation_label_field = AdjacencyField(
            indices=relation_indices, sequence_field=span_field, labels=relations, label_namespace="relation_labels"
        )

        # Pull it  all together.
        fields = dict(
            text=text_field,
            spans=span_field,
            ner_labels=ner_label_field,
            coref_labels=coref_label_field,
            relation_labels=relation_label_field,
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
                for span, span_label in zip(ins.fields["spans"].field_list, ins.fields['ner_labels'].labels)
                if span_label != ''
            ]

            return tokens, entities

        for key in instances_by_doc :
            instances_by_doc[key] = convert_list_to_structured(instances_by_doc[key])

        return instances_by_doc



