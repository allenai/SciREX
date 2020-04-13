import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from overrides import overrides

import numpy as np

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import TimeDistributed
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from dygie_pwc.training.ner_metrics import NERMetrics, NERMentionMetrics

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class SpanClassifier(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        mention_feedforward: FeedForward,
        label_namespace: str,
        balancing_strategy: str = "class_weight",
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super(SpanClassifier, self).__init__(vocab, regularizer)
        self._label_namespace = label_namespace
        # Number of classes determine the output dimension of the final layer
        self._n_labels = vocab.get_vocab_size(label_namespace)
        self._label_map = vocab.get_token_to_index_vocabulary(label_namespace)
        self._index_map = vocab.get_index_to_token_vocabulary(label_namespace)

        self.register_buffer("_class_weight", torch.Tensor([1.0] * self._n_labels))
        self.register_buffer("_sample_prob", torch.Tensor([1.0] * self._n_labels))
        self._registered_loss_modifiers = False

        self._mention_feedforward = TimeDistributed(mention_feedforward)
        self._ner_scorer = TimeDistributed(torch.nn.Linear(mention_feedforward.get_output_dim(), self._n_labels))
        self._ner_metrics = NERMetrics(self._label_map.keys())

        self._balancing_strategy = balancing_strategy

        initializer(self)

    @overrides
    def forward(
        self,  # type: ignore
        spans: torch.IntTensor,  # (Batch Size, Number of Spans, 2)
        span_mask: torch.IntTensor,  # (Batch Size, Number of Spans)
        span_embeddings: torch.IntTensor,  # (Batch Size, Number of Spans, Span Embedding SIze)
        ner_labels: torch.IntTensor = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:

        # Shape: (Batch_size, Number of spans, H)
        span_feedforward = self._mention_feedforward(span_embeddings)
        ner_scores = self._ner_scorer(span_feedforward)
        ner_probs = F.softmax(ner_scores, dim=-1)

        output_dict = {
            "spans" : spans,
            "span_mask" : span_mask,
            "ner_probs": ner_probs,
            "loss" : 0.0
        }

        if ner_labels is not None:
            ner_labels_str = self.map_combined_prediction_to_str(ner_labels.view(-1), span_mask)
            predicted_ner = self.predict_ner_from_combined(ner_scores, span_mask)
            self._ner_metrics(predicted_ner, ner_labels_str)
            loss = self._compute_loss_for_scores(ner_scores, ner_labels, span_mask, metadata)
            output_dict["loss"] = loss

        if metadata is not None:
            output_dict["metadata"] = metadata

        return output_dict

    def _compute_loss_for_scores(self, ner_scores, ner_labels, span_mask, metadata):
        mask_flat = span_mask.view(-1).bool()

        ner_scores_flat = ner_scores.view(-1, ner_scores.size(-1))[mask_flat]
        ner_labels_flat = ner_labels.view(-1)[mask_flat]

        if not self._registered_loss_modifiers:
            sample_prob = metadata[0][self._label_namespace + "_sample_prob"]
            for k, v in sample_prob.items():
                self._sample_prob[self._label_map[k]] = v

            class_weight = metadata[0][self._label_namespace + "_class_weight"]
            for k, v in class_weight.items():
                self._class_weight[self._label_map[k]] = v

            self._registered_loss_modifiers = True
            
        class_weight = None
        if self._balancing_strategy == 'sample' :            
            keep_element = torch.bernoulli(self._sample_prob[ner_labels_flat]).byte()
            ner_scores_flat = ner_scores_flat[keep_element]
            ner_labels_flat = ner_labels_flat[keep_element]
        elif self._balancing_strategy == 'class_weight' :
            class_weight = self._class_weight

        loss = torch.nn.CrossEntropyLoss(reduction="sum", weight=class_weight)(ner_scores_flat, ner_labels_flat)
        return loss

    def predict_ner_from_combined(self, ner_scores, span_mask):
        ner_scores_flat = ner_scores.view(-1, ner_scores.size(-1))
        _, predicted_ner = ner_scores_flat.max(-1)
        predicted_ner = self.map_combined_prediction_to_str(predicted_ner, span_mask)

        return predicted_ner

    def map_combined_prediction_to_str(self, predicted_ner, span_mask):
        mask_flat = span_mask.view(-1).byte()
        predicted_ner = predicted_ner[mask_flat]
        predicted_ner = list(map(lambda x: self._index_map[x], list(predicted_ner.cpu().data.numpy())))
        return predicted_ner

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):
        output_dict['decoded_spans'] = []
        if 'spans' in output_dict :
            for spans, spans_mask, spans_prob in zip(output_dict['spans'], output_dict['span_mask'], output_dict['ner_probs']) :
                spans_mask = spans_mask.byte()
                labels = spans_prob.argmax(-1)[spans_mask]
                spans = spans[spans_mask]
                decoded = {(span[0].item(), span[1].item() + 1): self._index_map[label.item()] for span, label in zip(spans, labels)}
                output_dict['decoded_spans'].append(decoded)

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self._ner_metrics.get_metric(reset)
        metrics = {self._label_namespace + '_' + k: v for k, v in metrics.items()}

        return metrics

