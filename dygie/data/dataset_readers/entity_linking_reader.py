from typing import Dict, List
import json
import logging

import numpy as np

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, MetadataField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer, Token
from itertools import combinations
from random import shuffle
from collections import Counter
from baseline.baseline import character_similarity_features

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("pwc_linker")
class PwCLinkerReader(DatasetReader):
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
        sample_train: bool = False,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy)
        self._sample_train = sample_train
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        pairs = self.generate_pairs(file_path)

        logger.info("NUMBER OF PAIRS - %d", len(pairs))

        c = Counter([x[2] for x in pairs])
        min_count = min(c.values())
        prob = {k: min(1, min_count / v) for k, v in c.items()}

        logger.info("Loaded all pairs from %s", file_path)
        for w1, w2, gold_label, t1, t2, features in pairs:
            yield self.text_to_instance(
                w1, w2, t1, t2, features, gold_label, prob[gold_label] if "train.jsonl" in file_path else 1.0
            )

    @staticmethod
    def generate_pairs(file_path):
        pairs = []
        with open(file_path, "r") as snli_file:
            logger.info("Reading Coref instances from jsonl dataset at: %s", file_path)
            for line in snli_file:
                ins = json.loads(line)
                entities = [(x[0], x[1], x[2].split("_")) for x in ins["ner"]]
                coref = {}
                for k, vlist in ins["coref"].items():
                    for v in vlist:
                        if tuple(v) not in coref:
                            coref[tuple(v)] = []
                        coref[tuple(v)].append(k)
                coref = {k: set(v) for k, v in coref.items()}
                # shuffle(entities)
                for e1, e2 in combinations(entities, 2):
                    if e1 == e2: 
                        continue
                    c1, c2 = coref.get((e1[0], e1[1]), set()), coref.get((e2[0], e2[1]), set())
                    t1, t2 = e1[2][0], e2[2][0]

                    if t1 != t2:
                        continue
                    w1, w2 = " ".join(ins["words"][e1[0] : e1[1]]), " ".join(ins["words"][e2[0] : e2[1]])
                    if w1.lower() == w2.lower() or len(c1 & c2) > 0:
                        gold_label = "Entailment"
                    elif len(c1) == 0 and len(c2) == 0:
                        gold_label = "-"
                    elif len(c1 & c2) == 0:
                        gold_label = "Contradiction"

                    if gold_label == "-":
                        continue

                    char_sim_features = character_similarity_features(w1, w2, max_ng=3)
                    features = np.array(
                        [(e1[1] - e2[0]) / len(ins["words"]), (e1[1] - e1[0]) / 10, (e2[1] - e2[0]) / 10] + char_sim_features
                    )
                    pairs.append((w1, w2, gold_label, t1, t2, features))
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
        prob: float = None,
    ) -> Instance:
        fields: Dict[str, Field] = {}
        premise_tokens = [Token(type_premise)] + self._tokenizer.tokenize(premise)
        hypothesis_tokens = [Token(type_hypothesis)] + self._tokenizer.tokenize(hypothesis)
        fields["premise"] = TextField(premise_tokens, self._token_indexers)
        fields["hypothesis"] = TextField(hypothesis_tokens, self._token_indexers)
        if label:
            fields["label"] = LabelField(label)

        fields["pair_features"] = ArrayField(features)

        metadata = {
            "premise_tokens": [x.text for x in premise_tokens],
            "hypothesis_tokens": [x.text for x in hypothesis_tokens],
            "keep_prob": prob,
        }
        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)


@DatasetReader.register("pwc_linker_bert")
class PwCLinkerReaderBERT(PwCLinkerReader):
    @overrides
    def text_to_instance(
        self,  # type: ignore
        premise: str,
        hypothesis: str,
        type_premise: str,
        type_hypothesis: str,
        features: List[float],
        label: str = None,
        prob: float = None,
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

        metadata = {
            "premise_tokens": [x.text for x in premise_tokens],
            "hypothesis_tokens": [x.text for x in hypothesis_tokens],
            "keep_prob": prob,
        }
        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)
