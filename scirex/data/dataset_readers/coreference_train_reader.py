import json
import logging
from collections import Counter
from itertools import combinations
from typing import Dict, List, Tuple

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, LabelField, MetadataField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("scirex_coreference_train_reader")
class ScirexCoreferenceTrainReader(DatasetReader):
    def __init__(
        self,
        sample_train: bool = False,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy)
        self._sample_train = sample_train
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        pairs = self.generate_pairs(file_path)

        logger.info("NUMBER OF PAIRS - %d", len(pairs))

        c = Counter([x[2] for x in pairs])
        min_count = min(c.values())
        prob = {k: min(1, min_count / v) for k, v in c.items()}

        logger.info("Loaded all pairs from %s", file_path)
        for w1, w2, gold_label in pairs:
            yield self.text_to_instance(
                w1, w2, gold_label, prob[gold_label] if "train.jsonl" in file_path else 1.0
            )

    @staticmethod
    def generate_pairs(file_path):
        pairs = []
        with open(file_path, "r") as data_file:
            for _, line in enumerate(data_file):
                ins = json.loads(line)
                entities: List[Tuple[int, int, str]] = [tuple(x) for x in ins["ner"]]

                clusters = {}
                for k, vlist in ins["coref"].items():
                    for v in vlist:
                        if tuple(v) not in clusters:
                            clusters[tuple(v)] = []
                        clusters[tuple(v)].append(k)

                clusters = {k: set(v) for k, v in clusters.items()}

                for mention_1, mention_2 in combinations(entities, 2):
                    type_1, type_2 = mention_1[2], mention_2[2]
                    if type_1 != type_2:
                        continue

                    cluster_labels_1, cluster_labels_2 = (
                        clusters.get((mention_1[0], mention_1[1]), set()),
                        clusters.get((mention_2[0], mention_2[1]), set()),
                    )
                    w1, w2 = (
                        " ".join(ins["words"][mention_1[0] : mention_1[1]]),
                        " ".join(ins["words"][mention_2[0] : mention_2[1]]),
                    )

                    if w1.lower() == w2.lower() or len(cluster_labels_1 & cluster_labels_2) > 0:
                        gold_label = 1
                    elif len(cluster_labels_1) == 0 and len(cluster_labels_2) == 0:
                        continue
                    elif len(cluster_labels_1 & cluster_labels_2) == 0:
                        gold_label = 0

                    pairs.append((type_1 + " " + w1, type_2 + " " + w2, gold_label))
        return pairs

    @overrides
    def text_to_instance(
        self,  # type: ignore
        premise: str,
        hypothesis: str,
        label: int,
        prob: float = None,
    ) -> Instance:
        fields: Dict[str, Field] = {}
        premise_tokens = self._tokenizer.tokenize(premise)
        hypothesis_tokens = self._tokenizer.tokenize(hypothesis)

        fields["tokens"] = TextField(
            [Token("[CLS]")] + premise_tokens + [Token("[SEP]")] + hypothesis_tokens, self._token_indexers
        )
        
        fields["label"] = LabelField(label, skip_indexing=True)

        metadata = {
            "premise_tokens": [x.text for x in premise_tokens],
            "hypothesis_tokens": [x.text for x in hypothesis_tokens],
            "keep_prob": prob,
        }
        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)
