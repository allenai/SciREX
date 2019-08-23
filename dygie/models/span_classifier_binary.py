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
from dygie.training.thresholding_f1_metric import BinaryThresholdF1

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class SpanClassifier(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        mention_feedforward: FeedForward,
        label_namespace: str,
        balancing_strategy: str = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super(SpanClassifier, self).__init__(vocab, regularizer)
        self._label_namespace = label_namespace

        self.register_buffer("_class_weight", torch.Tensor([1.0]))
        self.register_buffer("_sample_prob", torch.Tensor([1.0]))
        self._registered_loss_modifiers = False

        self._mention_feedforward = TimeDistributed(mention_feedforward)
        self._ner_scorer = TimeDistributed(torch.nn.Linear(mention_feedforward.get_output_dim(), 1))
        self._ner_metrics = BinaryThresholdF1()

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
        ner_scores = self._ner_scorer(span_feedforward).squeeze(-1) #(B, NS)
        ner_probs = torch.sigmoid(ner_scores)

        output_dict = {
            "spans" : spans,
            "span_mask" : span_mask,
            "ner_probs": ner_probs,
            "loss" : 0.0
        }

        if ner_labels is not None:
            assert ner_probs.shape == ner_labels.shape and len(ner_probs.shape) == 2
            self._ner_metrics(ner_probs, ner_labels)
            loss = self._compute_loss_for_scores(ner_probs, ner_labels, span_mask, metadata)
            output_dict["loss"] = loss

        if metadata is not None:
            output_dict["metadata"] = metadata

        return output_dict

    def _compute_loss_for_scores(self, ner_probs, ner_labels, span_mask, metadata):
        mask_flat = span_mask.view(-1).byte()

        ner_probs_flat = ner_probs.view(-1)[mask_flat]
        ner_labels_flat = ner_labels.view(-1)[mask_flat]

        # if not self._registered_loss_modifiers:
        #     sample_prob = metadata[0]["ner_labels_sample_prob"]
        #     for k, v in sample_prob.items():
        #         self._sample_prob[self._label_map[k]] = v

        #     class_weight = metadata[0]["ner_labels_class_weight"]
        #     for k, v in class_weight.items():
        #         self._class_weight[self._label_map[k]] = v

        #     self._registered_loss_modifiers = True
            
        # class_weight = None
        # if self._balancing_strategy == 'sample' :            
        #     keep_element = torch.bernoulli(self._sample_prob[ner_labels_flat]).byte()
        #     ner_scores_flat = ner_scores_flat[keep_element]
        #     ner_labels_flat = ner_labels_flat[keep_element]
        # elif self._balancing_strategy == 'class_weight' :
        #     class_weight = self._class_weight

        loss = torch.nn.BCELoss(reduction="sum")(ner_probs_flat, ner_labels_flat.float())
        return loss

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):
        output_dict['decoded_spans'] = []
        if 'spans' in output_dict :
            for spans, spans_mask, spans_prob in zip(output_dict['spans'], output_dict['span_mask'], output_dict['ner_probs']) :
                spans_mask = spans_mask.byte()
                labels = spans_prob[spans_mask]
                spans = spans[spans_mask]
                decoded = {(span[0].item(), span[1].item() + 1): label.item() for span, label in zip(spans, labels)}
                output_dict['decoded_spans'].append(decoded)

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self._ner_metrics.get_metric(reset)
        metrics = {"span_" + k: v for k, v in metrics.items()}

        return metrics

