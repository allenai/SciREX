from typing import Dict, Tuple, Any, List
import json
import logging
import numpy as np
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, MetadataField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, Token
from allennlp.data.tokenizers import SpacyTokenizer as WordTokenizer
from itertools import combinations
from baseline.baseline import character_similarity_features


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class PwCLinkerAllPairsReader(DatasetReader):
    """
    Coped from SNLi reader Allennlp

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    """

    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        lazy: bool = False,
        sample_train: bool = False,
        type: str = None,
    ) -> None:
        super().__init__(lazy)
        self.label_space = (0, 1)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        pairs = self.generate_pairs(file_path)

        logger.info("Loaded all pairs from %s", file_path)
        for p in pairs:
            yield self.text_to_instance(*p)

    def generate_pairs(self, file_path):
        pairs = []
        with open(file_path, "r") as snli_file:
            logger.info("Reading Coref instances from jsonl dataset at: %s", file_path)
            for line in snli_file:
                ins = json.loads(line)
                for field in ["prediction", "gold"]:
                    entities = [
                        (x["span"][0], x["span"][1], x["label"].split("_")[0])
                        for x in ins[field]
                    ]
                    words = ins["words"]
                    for e1, e2 in combinations(entities, 2):
                        w1 = " ".join(words[e1[0] : e1[1]])
                        w2 = " ".join(words[e2[0] : e2[1]])
                        t1, t2 = e1[2], e2[2]
                        gold_label = None
                        if t1 == t2:
                            metadata = {
                                "span_premise": e1,
                                "span_hypothesis": e2,
                                "doc_id": ins["doc_id"],
                                "field": field,
                            }
                            char_sim_features = character_similarity_features(w1, w2, max_ng=3)
                            features = np.array(
                                [(e1[1] - e2[0]) / len(ins["words"]), (e1[1] - e1[0]) / 10, (e2[1] - e2[0]) / 10] + char_sim_features
                            )
                            pairs.append((w1, w2, t1, t2, features, gold_label, metadata))

        return pairs

    @overrides
    def text_to_instance(
        self,  # type: ignore
        premise: str,
        hypothesis: str,
        type_premise: str,
        type_hypothesis: str,
        features: List[float],
        label: str = None,
        metadata: Dict[str, Any] = None,
    ) -> Instance:
        fields: Dict[str, Field] = {}
        premise_tokens = [Token(type_premise)] + self._tokenizer.tokenize(premise)
        hypothesis_tokens = [Token(type_hypothesis)] + self._tokenizer.tokenize(hypothesis)

        fields["premise"] = TextField(premise_tokens, self._token_indexers)
        fields["hypothesis"] = TextField(hypothesis_tokens, self._token_indexers)
        if label:
            fields["label"] = LabelField(label)

        fields["pair_features"] = ArrayField(features)

        metadata.update(
            {
                "premise_tokens": [x.text for x in premise_tokens],
                "hypothesis_tokens": [x.text for x in hypothesis_tokens],
                "keep_prob": 1.0,
            }
        )
        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)

class PwCLinkerAllPairsReaderBERT(PwCLinkerAllPairsReader):
    @overrides
    def text_to_instance(
        self,  # type: ignore
        premise: str,
        hypothesis: str,
        type_premise: str,
        type_hypothesis: str,
        features: List[float],
        label: str = None,
        metadata: Dict[str, Any] = None,
    ) -> Instance:
        fields: Dict[str, Field] = {}
        premise_tokens = self._tokenizer.tokenize(type_premise + " " + premise)
        hypothesis_tokens = self._tokenizer.tokenize(type_hypothesis + " " + hypothesis)

        fields["tokens"] = TextField(
            [Token("[CLS]")] + premise_tokens + [Token("[SEP]")] + hypothesis_tokens, self._token_indexers
        )
        if label:
            fields["label"] = LabelField(label)

        fields["pair_features"] = ArrayField(features)

        metadata.update(
            {
                "premise_tokens": [x.text for x in premise_tokens],
                "hypothesis_tokens": [x.text for x in hypothesis_tokens],
                "keep_prob": 1.0,
            }
        )
        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)