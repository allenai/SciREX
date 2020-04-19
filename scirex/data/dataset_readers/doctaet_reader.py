import json
import logging
from itertools import product
from typing import Dict, List, Tuple

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, MetadataField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token
from overrides import overrides
from scirex_utilities.entity_utils import used_entities

from scirex.data.utils.section_feature_extraction import filter_to_doctaet

Span = Tuple[int, int]
BaseEntityType = str
EntityType = Tuple[str, str]

map_label = lambda x, n: "_".join([x[i] for i in n]) if x != "" else ""
is_x_in_y = lambda x, y: x[0] >= y[0] and x[1] <= y[1]

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

does_overlap = lambda x, y: max(x[0], y[0]) < min(x[1], y[1])


def clean_json_dict(json_dict):
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
    sentences: List[List[Span]] = json_dict["sentences"]
    sections: List[Span] = json_dict["sections"]
    assert all(
        (sections[i][0] == sentences[i][0][0] and sections[i][-1] == sentences[i][-1][-1])
        for i in range(len(sections))
    ), breakpoint()


@DatasetReader.register("doctaet_reader")
class DoctaetReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None, lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers

    @overrides
    def _read(self, file_path: str):
        with open(file_path, "r") as g:
            for _, line in enumerate(g):
                json_dict = json.loads(line)
                json_dict = clean_json_dict(json_dict)
                verify_json_dict(json_dict)

                # Get fields from JSON dict
                doc_id = json_dict["doc_id"]
                doctaet_words = filter_to_doctaet(json_dict)

                n_ary_relations: List[Dict[BaseEntityType, str]] = json_dict["n_ary_relations"]

                possible_entities = {e: [] for e in used_entities}
                for relation in n_ary_relations:
                    for e in used_entities:
                        possible_entities[e].append(relation[e])

                possible_entities = {e: list(set(v)) for e, v in possible_entities.items()}
                candidate_relations = {}
                for possible_relation in product(*[possible_entities[e] for e in used_entities]):
                    candidate_relations[possible_relation] = False

                for relation in n_ary_relations:
                    candidate_relations[tuple([relation[e] for e in used_entities])] = True

                for candidate, label in candidate_relations.items():
                    instance = self.text_to_instance(
                        doc_id=doc_id,
                        document=doctaet_words,
                        relation=dict(zip(used_entities, candidate)),
                        label=label,
                    )
                    yield instance

    def text_to_instance(
        self, doc_id: str, document: List[str], relation: Dict[BaseEntityType, str], label: int
    ):

        relation_text = (" , ".join([relation[e] for e in used_entities]) + ' [SEP]').split()
        text_field = TextField([Token(word) for word in relation_text + document], self._token_indexers)
        metadata_field = MetadataField({"doc_id": doc_id, "relation": relation})
        label_field = LabelField(str(label))
        fields = dict(tokens=text_field, metadata=metadata_field, label=label_field)

        return Instance(fields)
