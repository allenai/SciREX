import logging
from typing import Any, Dict, List, Optional

import torch
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers.dataset_utils.span_utils import bioul_tags_to_spans
from allennlp.models.model import Model
from allennlp.modules import ConditionalRandomField, FeedForward, TimeDistributed
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from overrides import overrides

from dygie.training.span_f1_metrics import SpanBasedF1Measure

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

level_mapping = {1: [], 2: [(0,)], 3: [(0,), (0, 1), (0, 2)]}


class NERTagger(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        mention_feedforward: FeedForward,
        label_namespace: str,
        label_encoding: str = "BIOUL",
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super(NERTagger, self).__init__(vocab, regularizer)

        self._vocab = vocab
        self.label_namespace = label_namespace
        self._n_labels = vocab.get_vocab_size(label_namespace)

        self.label_map = self.vocab.get_index_to_token_vocabulary(label_namespace)
        print(self.label_map)

        self._mention_feedforward = TimeDistributed(mention_feedforward)

        self._ner_scorer = TimeDistributed(torch.nn.Linear(mention_feedforward.get_output_dim(), self._n_labels))
        constraints = allowed_transitions(label_encoding, self.vocab.get_index_to_token_vocabulary(label_namespace))
        self._ner_crf = ConditionalRandomField(self._n_labels, constraints, include_start_end_transitions=False)

        self._ner_metrics = SpanBasedF1Measure(self.label_map, label_encoding=label_encoding)

        levels = list(
            set(
                [
                    tuple(label.split("-")[1].split("_"))
                    for label in list(self.label_map.values())
                    if label.startswith("B-")
                ]
            )
        )
        if len(set([len(x) for x in list(levels)])) > 1:
            raise ConfigurationError("More than one level in NER labels {}".format(levels))

        self._ner_level_metrics = {}

        select_level = (
            lambda x, n: x.split("-")[0] + "-" + "_".join([x.split("-")[1].split("_")[i] for i in n])
            if x != "O"
            else "O"
        )

        for level in level_mapping[len(levels[0])]:
            self._ner_level_metrics[level] = SpanBasedF1Measure(
                label_vocabulary={k: select_level(v, level) for k, v in self.label_map.items()},
                label_encoding=label_encoding,
            )

        initializer(self)

    @overrides
    def forward(
        self,  # type: ignore
        text_embeddings: torch.FloatTensor,
        text_mask: torch.IntTensor,
        ner_labels: torch.IntTensor = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:

        # Shape: (Batch_size, Number of spans, H)
        span_feedforward = self._mention_feedforward(text_embeddings)
        ner_scores = self._ner_scorer(span_feedforward)
        predicted_ner = self._ner_crf.viterbi_tags(ner_scores, text_mask)

        predicted_ner = [x for x, y in predicted_ner]
        gold_ner = [list(x[m.bool()].detach().cpu().numpy()) for x, m in zip(ner_labels, text_mask)]

        output = {"logits": ner_scores, "tags": predicted_ner, "gold_tags": gold_ner}

        if ner_labels is not None:
            # Add negative log-likelihood as loss
            try :
                log_likelihood = self._ner_crf(ner_scores, ner_labels, text_mask)
            except :
                breakpoint()
            output["loss"] = -log_likelihood / text_embeddings.shape[0]

            # Represent viterbi tags as "class probabilities" that we can
            # feed into the metrics
            class_probabilities = ner_scores * 0.0
            for i, instance_tags in enumerate(predicted_ner):
                for j, tag_id in enumerate(instance_tags):
                    if i >= ner_scores.shape[0] or j >= ner_scores.shape[1] or tag_id >= ner_scores.shape[2]:
                        breakpoint()
                    class_probabilities[i, j, tag_id] = 1

            self._ner_metrics(class_probabilities, ner_labels, text_mask.float())
            for level in self._ner_level_metrics:
                self._ner_level_metrics[level](class_probabilities, ner_labels, text_mask.float())

        if metadata is not None:
            output["metadata"] = metadata

        return output

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):
        output_dict["decoded_ner"] = []
        output_dict["gold_ner"] = []

        for instance_tags in output_dict["tags"]:
            typed_spans = bioul_tags_to_spans(
                [self.vocab.get_token_from_index(tag, namespace=self.label_namespace) for tag in instance_tags]
            )
            output_dict["decoded_ner"].append({(x[1][0], x[1][1] + 1): x[0] for x in typed_spans})

        for instance_tags in output_dict["gold_tags"]:
            typed_spans = bioul_tags_to_spans(
                [self.vocab.get_token_from_index(tag, namespace=self.label_namespace) for tag in instance_tags]
            )
            output_dict["gold_ner"].append({(x[1][0], x[1][1] + 1): x[0] for x in typed_spans})

        output_dict = self._extract_spans(output_dict)

        return output_dict

    def _extract_spans(self, output_dict: Dict[str, torch.Tensor]):
        predicted_spans = [[(k[0], k[1] - 1, v) for k, v in doc.items()] for doc in output_dict["decoded_ner"]]
        max_spans_count = max([len(s) for s in predicted_spans])

        for spans in predicted_spans:
            for _ in range(max_spans_count - len(spans)):
                spans.append((-1, -1, "Entity_Method"))

        spans_lists = [[span[:2] for span in spans] for spans in predicted_spans]
        entity_label_map = self._vocab.get_token_to_index_vocabulary("span_entity_labels")
        span_labels_lists = [[entity_label_map[span[2].split("_")[1]] for span in spans] for spans in predicted_spans]

        output_dict["spans"] = torch.LongTensor(spans_lists)
        output_dict["span_labels"] = torch.LongTensor(span_labels_lists)

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self._ner_metrics.get_metric(reset)
        metrics = {"ner_" + k.replace("-overall", ""): v for k, v in metrics.items()}

        for level, level_metrics in self._ner_level_metrics.items():
            level_metrics = level_metrics.get_metric(reset)
            level_metrics = {
                "ner_" + str(len(level)) + "_" + k.replace("-overall", ""): v for k, v in level_metrics.items()
            }
            metrics.update(level_metrics)

        return metrics
