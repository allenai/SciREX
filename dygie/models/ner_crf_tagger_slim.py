import logging
from typing import Any, Dict, List, Optional

import torch
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions

from allennlp.modules import TimeDistributed
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator

from allennlp.data.dataset_readers.dataset_utils.span_utils import *

from dygie.training.span_f1_metrics import SpanBasedF1Measure

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

level_mapping = {1: [], 2: [(0,)], 3: [(0,), (0, 1), (0, 2)]}


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
        label_encoding: str = "BIOUL",
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super(NERTagger, self).__init__(vocab, regularizer)

        self._vocab = vocab
        self.label_namespace = label_namespace
        self._n_labels = vocab.get_vocab_size(label_namespace)
        # self.register_buffer(label_namespace + '_weight', torch.Tensor([0.]*self._n_labels))

        self.label_map = self.vocab.get_index_to_token_vocabulary(label_namespace)
        self.index_map = self.vocab.get_token_to_index_vocabulary(label_namespace)

        print(self.label_map)
        self.O_index = self.index_map["O"]

        self._registered_class_weight = False

        self._mention_feedforward = TimeDistributed(mention_feedforward)

        self._ner_scorer = TimeDistributed(torch.nn.Linear(mention_feedforward.get_output_dim(), self._n_labels))
        constraints = allowed_transitions(label_encoding, self.vocab.get_index_to_token_vocabulary(label_namespace))
        self._ner_crf = ConditionalRandomField(self._n_labels, constraints, include_start_end_transitions=False)

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
        gold_ner = [list(x[m.byte()].detach().cpu().numpy()) for x, m in zip(ner_labels, text_mask)]

        output = {"logits": ner_scores, "tags": predicted_ner, "gold_tags": gold_ner}

        if ner_labels is not None:
            # Add negative log-likelihood as loss
            log_likelihood = self._ner_crf(ner_scores, ner_labels, text_mask)
            output["loss"] = -log_likelihood #+ (torch.nn.CrossEntropyLoss(reduction="none")(
            #     ner_scores.view(-1, ner_scores.shape[-1]), ner_labels.view(-1)
            # ) * (0.5 * (ner_labels.view(-1) == self.O_index).float() + (ner_labels.view(-1) != self.O_index).float())).mean()

        if metadata is not None:
            output["metadata"] = metadata

        return output

    @staticmethod
    def _append_dummy_scores(ner_scores):
        dummy_scores = ner_scores.new_zeros(ner_scores.size(0), ner_scores.size(1), 1)
        ner_scores = torch.cat((dummy_scores, ner_scores), -1)
        return ner_scores

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):
        output_dict["decoded_ner"] = []
        output_dict["gold_ner"] = []
        for instance_tags in output_dict["tags"]:
            typed_spans = bioul_tags_to_spans(
                [self.vocab.get_token_from_index(tag, namespace=self.label_namespace) for tag in instance_tags]
            )
            output_dict["decoded_ner"].append({(x[1][0], x[1][1] + 1): x[0].split("_") for x in typed_spans})

        for instance_tags in output_dict["gold_tags"]:
            typed_spans = bioul_tags_to_spans(
                [self.vocab.get_token_from_index(tag, namespace=self.label_namespace) for tag in instance_tags]
            )
            output_dict["gold_ner"].append({(x[1][0], x[1][1] + 1): x[0].split("_") for x in typed_spans})

        predicted_spans = [[(k[0], k[1] - 1, v) for k, v in doc.items()] for doc in output_dict["decoded_ner"]]
        max_spans_count = max([len(s) for s in predicted_spans])

        for spans in predicted_spans:
            for _ in range(max_spans_count - len(spans)):
                spans.append((-1, -1, ""))

        spans_lists = [[span[:2] for span in spans] for spans in predicted_spans]
        span_labels_lists = [[span[2] for span in spans] for spans in predicted_spans]

        output_dict["spans"] = torch.LongTensor(spans_lists)
        output_dict["span_labels"] = span_labels_lists

        return output_dict
