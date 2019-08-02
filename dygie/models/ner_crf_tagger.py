import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions

from allennlp.modules import TimeDistributed
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator

from allennlp.data.dataset_readers.dataset_utils.span_utils import *

from allennlp.training.metrics import SpanBasedF1Measure

from math import log

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class NERTagger(Model):
    """
    Named entity recognition module of DyGIE model.

    Parameters
    ----------
    mention_feedforward : ``FeedForward``
        This feedforward network is applied to the span representations which is then scored
        by a linear layer.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        mention_feedforward: FeedForward,
        label_namespace: str,
        label_encoding: str = 'BIOUL',
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super(NERTagger, self).__init__(vocab, regularizer)

        self.label_namespace = label_namespace
        # Number of classes determine the output dimension of the final layer
        self._n_labels = vocab.get_vocab_size(label_namespace)
        self.register_buffer(self.label_namespace + "_weight", torch.Tensor([1.0] * self._n_labels))

        self.label_map = self.vocab.get_index_to_token_vocabulary(label_namespace)

        self._registered_class_weight = False

        self._mention_feedforward = TimeDistributed(mention_feedforward)

        self._ner_scorer = TimeDistributed(torch.nn.Linear(mention_feedforward.get_output_dim(), self._n_labels))
        constraints = allowed_transitions(label_encoding, self.vocab.get_index_to_token_vocabulary("ner_labels"))
        self._ner_crf = ConditionalRandomField(self._n_labels, constraints, include_start_end_transitions=False)

        self._ner_metrics = SpanBasedF1Measure(self.vocab, self.label_namespace, label_encoding=label_encoding)

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

        output = {"logits": ner_scores, "tags": predicted_ner}

        if ner_labels is not None:
            # Add negative log-likelihood as loss
            log_likelihood = self._ner_crf(ner_scores, ner_labels, text_mask)
            output["loss"] = -log_likelihood

            # Represent viterbi tags as "class probabilities" that we can
            # feed into the metrics
            class_probabilities = ner_scores * 0.
            for i, instance_tags in enumerate(predicted_ner):
                for j, tag_id in enumerate(instance_tags):
                    class_probabilities[i, j, tag_id] = 1

            self._ner_metrics(class_probabilities, ner_labels, text_mask.float())

        if metadata is not None:
            output["words"] = [x["sentence"] for x in metadata]
        return output

    def map_weights_to_tensor(self, metadata):
        weight = getattr(self, self.label_namespace + '_weight')
        if self._registered_class_weight is False:
            weight_dict = metadata[self.label_namespace + "_class_weight"]
            labels = self.vocab.get_token_to_index_vocabulary(self.label_namespace)
            for k, v in labels.items():
                if k in weight_dict:
                    weight[v] = (0.6 + 2 * weight_dict[k] / 5) if weight_dict[k] < 1 else log(weight_dict[k])
                else:
                    weight[v] = 0.0
            print("SETTING Class weight for " + self.label_namespace, weight)
            self._registered_class_weight = True

        return weight

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):
        output_dict["tags"] = [
            [self.vocab.get_token_from_index(tag, namespace=self.label_namespace)
                for tag in instance_tags]
            for instance_tags in output_dict["tags"]
        ]

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self._ner_metrics.get_metric(reset)
        metrics = {"ner_" + k.replace('-overall', ''): v for k, v in metrics.items()}
        return metrics
