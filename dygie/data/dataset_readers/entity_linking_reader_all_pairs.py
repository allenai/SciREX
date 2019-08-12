from typing import Dict, Tuple, Any
import json
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from itertools import combinations
from random import shuffle
import random
from collections import Counter

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
        type: str = None
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
            yield self.text_to_instance(p[0], p[1], p[2], p[3])

    def generate_pairs(self, file_path):
        pairs = []
        with open(file_path, "r") as snli_file:
            logger.info("Reading Coref instances from jsonl dataset at: %s", file_path)
            for line in snli_file:
                ins = json.loads(line)
                entities = [
                    (x["span"][0], x["span"][1], tuple(x["label"].split("_")[i] for i in self.label_space))
                    for x in ins["prediction"]
                ]
                words = ins["words"]
                for e1, e2 in combinations(entities, 2):
                    w1 = " ".join(words[e1[0] : e1[1]])
                    w2 = " ".join(words[e2[0] : e2[1]])
                    gold_label = None
                    if e1[2] == e2[2]:
                        metadata = {"span_premise": e1, "span_hypothesis": e2, "doc_id": ins["doc_id"]}
                        pairs.append((w1, w2, gold_label, metadata))

        return pairs

    @overrides
    def text_to_instance(
        self,  # type: ignore
        premise: str,
        hypothesis: str,
        label: str = None,
        metadata: Dict[str, Any] = None,
    ) -> Instance:
        fields: Dict[str, Field] = {}
        premise_tokens = self._tokenizer.tokenize(premise)
        hypothesis_tokens = self._tokenizer.tokenize(hypothesis)
        fields["premise"] = TextField(premise_tokens, self._token_indexers)
        fields["hypothesis"] = TextField(hypothesis_tokens, self._token_indexers)
        if label:
            fields["label"] = LabelField(label)

        metadata.update(
            {
                "premise_tokens": [x.text for x in premise_tokens],
                "hypothesis_tokens": [x.text for x in hypothesis_tokens],
                "keep_prob" : 1.0
            }
        )
        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)
