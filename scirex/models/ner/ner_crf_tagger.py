from typing import Any, Dict, List, Optional

import torch
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers.dataset_utils.span_utils import bioul_tags_to_spans
from allennlp.models.model import Model
from allennlp.modules import ConditionalRandomField, FeedForward, TimeDistributed
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from overrides import overrides

from scirex.metrics.span_f1_metrics import SpanBasedF1Measure
from allennlp.training.metrics.span_based_f1_measure import SpanBasedF1Measure as SpanBasedF1MeasureAllennlp


class NERTagger(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        mention_feedforward: FeedForward,
        label_namespace: str = "ner_type_labels",
        label_encoding: str = "BIOUL",
        exact_match: bool = False,
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

        self._ner_scorer = TimeDistributed(
            torch.nn.Linear(mention_feedforward.get_output_dim(), self._n_labels)
        )
        constraints = allowed_transitions(
            label_encoding, self.vocab.get_index_to_token_vocabulary(label_namespace)
        )
        self._ner_crf = ConditionalRandomField(
            self._n_labels, constraints, include_start_end_transitions=False
        )

        if exact_match:
            self._ner_metrics = SpanBasedF1Measure(self.label_map, label_encoding=label_encoding)
        else:
            self._ner_metrics = SpanBasedF1MeasureAllennlp(
                vocabulary=vocab, tag_namespace=label_namespace, label_encoding=label_encoding
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
            log_likelihood = self._ner_crf(ner_scores, ner_labels, text_mask)
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

        if metadata is not None:
            output["metadata"] = metadata

        return output

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):
        output_dict["decoded_ner"] = []
        output_dict["gold_ner"] = []

        for instance_tags in output_dict["tags"]:
            typed_spans = bioul_tags_to_spans(
                [
                    self.vocab.get_token_from_index(tag, namespace=self.label_namespace)
                    for tag in instance_tags
                ]
            )
            output_dict["decoded_ner"].append({(x[1][0], x[1][1] + 1): x[0] for x in typed_spans})

        for instance_tags in output_dict["gold_tags"]:
            typed_spans = bioul_tags_to_spans(
                [
                    self.vocab.get_token_from_index(tag, namespace=self.label_namespace)
                    for tag in instance_tags
                ]
            )
            output_dict["gold_ner"].append({(x[1][0], x[1][1] + 1): x[0] for x in typed_spans})

        output_dict = self._extract_spans(output_dict)

        return output_dict

    def _extract_spans(self, output_dict: Dict[str, torch.Tensor]):
        predicted_spans = [
            [(k[0], k[1] - 1, v) for k, v in doc.items()] for doc in output_dict["decoded_ner"]
        ]
        max_spans_count = max([len(s) for s in predicted_spans])

        for spans in predicted_spans:
            for _ in range(max_spans_count - len(spans)):
                spans.append((-1, -1, "Method"))

        spans_lists = [[span[:2] for span in spans] for spans in predicted_spans]
        entity_label_map = self._vocab.get_token_to_index_vocabulary("span_type_labels")
        span_labels_lists = [
            [entity_label_map[span[2]] for span in spans] for spans in predicted_spans
        ]

        output_dict["spans"] = torch.LongTensor(spans_lists)
        output_dict["span_labels"] = torch.LongTensor(span_labels_lists)

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self._ner_metrics.get_metric(reset)
        metrics = {"ner_" + k.replace("-overall", ""): v for k, v in metrics.items()}

        return metrics
