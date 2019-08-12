from typing import Dict
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

    def __init__(self,
                 sample_train: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._sample_train = sample_train
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        pairs = self.generate_pairs(file_path)

        c = Counter([x[2] for x in pairs])
        min_count = min(c.values())
        prob = {k:min(1, min_count / v) for k, v in c.items()}

        logger.info("Loaded all pairs from %s", file_path)
        for w1, w2, gold_label in pairs :
            yield self.text_to_instance(w1, w2, gold_label, prob[gold_label] if 'train.jsonl' in file_path else 1.0)

    @staticmethod
    def generate_pairs(file_path):
        pairs = []
        with open(file_path, 'r') as snli_file:
            logger.info("Reading Coref instances from jsonl dataset at: %s", file_path)
            i = 0
            for line in snli_file:
                ins = json.loads(line)
                entities = [(x[0], x[1]) for x in ins["ner"]]
                coref = {}
                for k, vlist in ins["coref"].items():
                    for v in vlist :
                        if tuple(v) not in coref:
                            coref[tuple(v)] = []
                        coref[tuple(v)].append(k)
                coref = {k: set(v) for k, v in coref.items()}
                for e1, e2 in combinations(entities, 2):
                    a = [e1, e2]
                    shuffle(a)
                    e1, e2 = a
                    c1, c2 = coref.get(e1, set()), coref.get(e2, set())
                    w1, w2 = " ".join(ins["words"][e1[0]:e1[1]]), " ".join(ins["words"][e2[0]:e2[1]])
                    if w1.lower() == w2.lower() or len(c1 & c2) > 0:
                        gold_label = 'Entailment'
                    elif len(c1) == 0 and len(c2) == 0:
                        gold_label = '-'
                    elif len(c1 & c2) == 0:
                        gold_label = 'Contradiction'

                    if gold_label == '-':
                        continue

                    pairs.append((w1, w2, gold_label))
        return pairs

    @overrides
    def text_to_instance(self,  # type: ignore
                         premise: str,
                         hypothesis: str,
                         label: str = None,
                         prob: float = None) -> Instance:
        fields: Dict[str, Field] = {}
        premise_tokens = self._tokenizer.tokenize(premise)
        hypothesis_tokens = self._tokenizer.tokenize(hypothesis)
        fields['premise'] = TextField(premise_tokens, self._token_indexers)
        fields['hypothesis'] = TextField(hypothesis_tokens, self._token_indexers)
        if label:
            fields['label'] = LabelField(label)

        metadata = {"premise_tokens": [x.text for x in premise_tokens],
                    "hypothesis_tokens": [x.text for x in hypothesis_tokens],
                    "keep_prob" : prob}
        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)