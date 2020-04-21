import json
import logging
from itertools import combinations
from typing import Any, Dict, Tuple

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import MetadataField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import SpacyTokenizer as WordTokenizer
from allennlp.data.tokenizers import Token, Tokenizer
from overrides import overrides
from tqdm import tqdm

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class ScirexCoreferenceEvalReader(DatasetReader):
    def __init__(
        self,
        field: str,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy)
        self._field = field
        self.label_space = (0, 1)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        pairs = self.generate_pairs(file_path)

        logger.info("Loaded all pairs from %s", file_path)
        for p in pairs:
            yield self.text_to_instance(*p)

    def generate_pairs(self, file_path):
        pairs = []
        with open(file_path, "r") as data_file:
            for line in tqdm(data_file):
                ins = json.loads(line)
                if self._field not in ins:
                    continue
                entities: Tuple[int, int, str] = ins[self._field]
                words = ins["words"]
                for e1, e2 in combinations(entities, 2):
                    w1 = " ".join(words[e1[0] : e1[1]])
                    w2 = " ".join(words[e2[0] : e2[1]])
                    t1, t2 = e1[2], e2[2]
                    if t1 == t2:
                        metadata = {
                            "span_premise": e1,
                            "span_hypothesis": e2,
                            "doc_id": ins["doc_id"],
                            "field": self._field,
                        }
                        pairs.append((t1 + " " + w1, t2 + " " + w2, metadata))

        print(len(pairs))

        return pairs

    @overrides
    def text_to_instance(
        self,  # type: ignore
        premise: str,
        hypothesis: str,
        metadata: Dict[str, Any] = None,
    ) -> Instance:
    
        fields = {}
        premise_tokens = self._tokenizer.tokenize(premise)
        hypothesis_tokens = self._tokenizer.tokenize(hypothesis)

        fields["tokens"] = TextField(
            [Token("[CLS]")] + premise_tokens + [Token("[SEP]")] + hypothesis_tokens, self._token_indexers
        )

        metadata.update(
            {
                "premise_tokens": [x.text for x in premise_tokens],
                "hypothesis_tokens": [x.text for x in hypothesis_tokens],
                "keep_prob": 1.0,
            }
        )
        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)
