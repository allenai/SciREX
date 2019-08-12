import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import TimeDistributed
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from dygie.training.ner_metrics import NERMetrics

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
        split_prediction: bool = False,
        decoding_type: str = "dp_decode",
        decoding_metric: str = "entropy",
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super(NERTagger, self).__init__(vocab, regularizer)

        self._split_prediction = split_prediction

        # Number of classes determine the output dimension of the final layer
        self._n_labels = vocab.get_vocab_size("ner_labels")
        self._n_entity_labels = vocab.get_vocab_size("ner_entity_labels")
        self._n_link_labels = vocab.get_vocab_size("ner_link_labels")

        self.register_buffer("ner_link_labels_weight", torch.Tensor([1.0] * self._n_link_labels))
        self.register_buffer("ner_entity_labels_weight", torch.Tensor([1.0] * self._n_entity_labels))
        self.register_buffer("ner_labels_weight", torch.Tensor([1.0] * self._n_labels))

        self._registered_class_weight = {}
        for label_namespace in ["ner_labels", "ner_entity_labels", "ner_link_labels"]:
            self._registered_class_weight[label_namespace] = False

        if not self._split_prediction:
            self._linked_tokens = [
                v for k, v in vocab.get_token_to_index_vocabulary("ner_labels").items() if "True" in k
            ]
            print("The Linked tokens are ", self._linked_tokens)

        # Null label is needed to keep track of when calculating the metrics
        null_label = vocab.get_token_index("", "ner_labels")
        assert null_label == 0  # If not, the dummy class won't correspond to the null label.
        assert vocab.get_token_index("", "ner_entity_labels") == 0
        assert vocab.get_token_index("", "ner_link_labels") == 0

        self._mention_feedforward = TimeDistributed(mention_feedforward)

        if self._split_prediction:
            self._ner_entity_scorer = TimeDistributed(
                torch.nn.Linear(mention_feedforward.get_output_dim(), self._n_entity_labels - 1)
            )
            self._ner_link_scorer = TimeDistributed(
                torch.nn.Linear(mention_feedforward.get_output_dim() + self._n_entity_labels, self._n_link_labels - 1)
            )
        else:
            self._ner_scorer = TimeDistributed(
                torch.nn.Linear(mention_feedforward.get_output_dim(), self._n_labels - 1)
            )

        self._ner_metrics = NERMetrics(list(vocab.get_token_to_index_vocabulary("ner_entity_labels").keys()))

        self._decoding_metric = decoding_metric
        self._decoding_type = decoding_type

        initializer(self)

    @overrides
    def forward(
        self,  # type: ignore
        spans: torch.IntTensor,  # (Batch Size, Number of Spans, 2)
        span_mask: torch.IntTensor,  # (Batch Size, Number of Spans)
        span_embeddings: torch.IntTensor,  # (Batch Size, Number of Spans, Span Embedding SIze)
        ner_labels: torch.IntTensor = None,
        ner_entity_labels: torch.IntTensor = None,
        ner_link_labels: torch.IntTensor = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:

        # Shape: (Batch_size, Number of spans, H)
        span_feedforward = self._mention_feedforward(span_embeddings)
        if self._split_prediction:
            ner_entity_scores = self._append_dummy_scores(self._ner_entity_scorer(span_feedforward))
            ner_link_scores = self._append_dummy_scores(
                self._ner_link_scorer(torch.cat([span_feedforward, ner_entity_scores], -1))
            )
            predicted_ner = self.predict_ner_from_separate(ner_entity_scores, ner_link_scores, span_mask)
        else:
            ner_scores = self._append_dummy_scores(self._ner_scorer(span_feedforward))
            predicted_ner = self.predict_ner_from_combined(ner_scores, span_mask)

        if self._split_prediction:
            ner_linked_scores_out = F.softmax(ner_link_scores, dim=-1)[:, :, 1]
        else:
            ner_linked_scores_out = F.softmax(ner_scores, dim=-1)[:, :, self._linked_tokens].sum(-1, keepdim=True)

        ner_scores_out = ner_scores if not self._split_prediction else ner_entity_scores
        ner_scores_out = F.softmax(ner_scores_out[:, :, 1:], dim=-1)
        output_dict = {
            "spans": spans,
            "span_mask": span_mask,
            "ner_scores": ner_scores_out,
            "predicted_ner": predicted_ner,
            "ner_linked_scores": ner_linked_scores_out,
        }

        if ner_labels is not None:
            ner_labels_str = self.map_combined_prediction_to_str(ner_labels.view(-1), span_mask)
            self._ner_metrics(predicted_ner, ner_labels_str)

            if self._split_prediction:

                loss = self._compute_loss_for_scores(
                    ner_entity_scores,
                    ner_entity_labels,
                    span_mask,
                    weight=self.map_weights_to_tensor(metadata[0], "ner_entity_labels"),
                ) + self._compute_loss_for_scores(
                    ner_link_scores,
                    ner_link_labels,
                    span_mask,
                    weight=self.map_weights_to_tensor(metadata[0], "ner_link_labels"),
                )
            else:
                loss = self._compute_loss_for_scores(
                    ner_scores,
                    ner_labels,
                    span_mask,
                    weight=self.map_weights_to_tensor(metadata[0], "ner_labels"),
                )

            output_dict["loss"] = loss

        if metadata is not None:
            output_dict["document"] = [x["sentence"] for x in metadata]

        return output_dict

    @staticmethod
    def _compute_loss_for_scores(ner_scores, ner_labels, span_mask, weight=None):
        ner_scores_flat = ner_scores.view(-1, ner_scores.size(-1))
        ner_labels_flat = ner_labels.view(-1)
        mask_flat = span_mask.view(-1).byte()

        loss = torch.nn.CrossEntropyLoss(reduction="sum", weight=weight)(
            ner_scores_flat[mask_flat], ner_labels_flat[mask_flat]
        )
        return loss

    @staticmethod
    def _append_dummy_scores(ner_scores):
        dummy_scores = ner_scores.new_zeros(ner_scores.size(0), ner_scores.size(1), 1)
        ner_scores = torch.cat((dummy_scores, ner_scores), -1)
        return ner_scores

    def predict_ner_from_separate(self, ner_entity_scores, ner_link_scores, span_mask):
        mask_flat = span_mask.view(-1).byte()
        map_index_to_label = self.vocab.get_index_to_token_vocabulary("ner_entity_labels")
        _, predicted_entity = ner_entity_scores.view(-1, ner_entity_scores.size(-1)).max(-1)
        predicted_entity = list(
            map(lambda x: map_index_to_label[x], list(predicted_entity[mask_flat].cpu().data.numpy()))
        )

        map_index_to_label = self.vocab.get_index_to_token_vocabulary("ner_link_labels")
        _, predicted_link = ner_link_scores.view(-1, ner_link_scores.size(-1)).max(-1)
        predicted_link = list(map(lambda x: map_index_to_label[x], list(predicted_link[mask_flat].cpu().data.numpy())))

        predicted_ner = [
            "" if x == "" else (x + "_" + (y if y != "" else "False")) for x, y in zip(predicted_entity, predicted_link)
        ]
        return predicted_ner

    def predict_ner_from_combined(self, ner_scores, span_mask):
        ner_scores_flat = ner_scores.view(-1, ner_scores.size(-1))
        _, predicted_ner = ner_scores_flat.max(-1)
        predicted_ner = self.map_combined_prediction_to_str(predicted_ner, span_mask)

        return predicted_ner

    def map_combined_prediction_to_str(self, predicted_ner, span_mask):
        mask_flat = span_mask.view(-1).byte()
        predicted_ner = predicted_ner[mask_flat]
        map_index_to_label = self.vocab.get_index_to_token_vocabulary("ner_labels")
        predicted_ner = list(map(lambda x: map_index_to_label[x], list(predicted_ner.cpu().data.numpy())))
        return predicted_ner

    def map_weights_to_tensor(self, metadata, label_namespace: str):
        weight = getattr(self, label_namespace + "_weight")
        if self._registered_class_weight[label_namespace] is False:
            weight_dict = metadata[label_namespace + "_class_weight"]
            labels = self.vocab.get_token_to_index_vocabulary(label_namespace)
            for k, v in labels.items():
                if k in weight_dict:
                    weight[v] = (0.6 + 2*weight_dict[k]/5) if weight_dict[k] < 1 else log(weight_dict[k])
                else:
                    weight[v] = 0.0
            print("SETTING Class weight for " + label_namespace, weight)
            self._registered_class_weight[label_namespace] = True

        return weight

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):
        return getattr(self, self._decoding_type)(output_dict)

    def all_decode(self, output_dict: Dict[str, torch.Tensor]):
        predicted_ner_batch = output_dict["predicted_ner"].detach().cpu()
        spans_batch = output_dict["spans"].detach().cpu()
        span_mask_batch = output_dict["span_mask"].detach().cpu().byte()

        res_list = []
        res_dict = []
        for spans, span_mask, predicted_NERs in zip(spans_batch, span_mask_batch, predicted_ner_batch):
            entry_list = []
            entry_dict = {}
            for span, ner in zip(spans[span_mask], predicted_NERs[span_mask]):
                ner = ner.item()
                if ner > 0:
                    the_span = (span[0].item(), span[1].item())
                    the_label = self.vocab.get_token_from_index(ner, "ner_labels")
                    entry_list.append((the_span[0], the_span[1], the_label))
                    entry_dict[the_span] = the_label
            res_list.append(entry_list)
            res_dict.append(entry_dict)

        output_dict["decoded_ner"] = res_list
        output_dict["decoded_ner_dict"] = res_dict
        return output_dict

    def dp_decode(self, output_dict: Dict[str, torch.Tensor]):
        dp_scores = getattr(self, self._decoding_metric)(output_dict["ner_scores"])
        predicted_ner_batch = output_dict["predicted_ner"].detach().cpu()
        spans_batch = output_dict["spans"].detach().cpu()
        span_mask_batch = output_dict["span_mask"].detach().cpu().byte()

        res_list = []
        res_dict = []
        for spans, spans_mask, spans_predicted_ner, spans_dp_scores in zip(
            spans_batch, span_mask_batch, predicted_ner_batch, dp_scores
        ):
            entry_dict = {}
            spans, spans_predicted_ner, spans_dp_scores = (
                spans[spans_mask],
                spans_predicted_ner[spans_mask],
                spans_dp_scores[spans_mask],
            )
            for span, ner_label, dp_score in zip(spans, spans_predicted_ner, spans_dp_scores):
                ner = ner_label.item()
                length, max_span_length = 0, 0
                the_span = (span[0].item(), span[1].item())
                length = max(length, the_span[1] + 1)
                max_span_length = max(max_span_length, the_span[1] - the_span[0] + 1)
                the_label = self.vocab.get_token_from_index(ner, "ner_labels")
                entry_dict[the_span] = (the_label, dp_score.item())

            entry_dict = self.run_dynamic_programming(entry_dict, length, max_span_length)
            res_list.append([(k[0], k[1], label) for k, label in entry_dict.items()])
            res_dict.append(entry_dict)

        output_dict["decoded_ner"] = res_list
        output_dict["decoded_ner_dict"] = res_dict
        return output_dict

    @staticmethod
    def run_dynamic_programming(entry_dict, doc_length, max_span_length):
        base_score = float("inf")
        doc_length_position = [-1] * doc_length
        doc_length_best_score = [base_score] * doc_length

        for i in range(doc_length):
            for j in range(1, max_span_length + 1):
                span_end = i
                span_start = span_end - j + 1
                if span_start < 0:
                    continue
                last_index = span_start - 1
                curr_span_score = entry_dict[(span_start, span_end)][1]
                last_index_score = doc_length_best_score[last_index] if last_index >= 0 else 0
                curr_score = last_index_score + curr_span_score
                if curr_score < doc_length_best_score[i]:
                    doc_length_best_score[i] = curr_score
                    doc_length_position[i] = last_index

        decoded_entry_dict = {}
        backlink = doc_length - 1
        while backlink >= 0:
            best_last_index = doc_length_position[backlink]
            span_start = best_last_index + 1
            span_end = backlink
            if entry_dict[(span_start, span_end)][0] > 0:
                decoded_entry_dict[(span_start, span_end)] = entry_dict[(span_start, span_end)][0]
            backlink = best_last_index

        return decoded_entry_dict

    def entropy(self, scores: torch.Tensor):
        p, log_p = F.softmax(scores, dim=-1), F.log_softmax(scores, dim=-1)
        h = -(p * log_p).sum(-1)
        return h

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self._ner_metrics.get_metric(reset)
        metrics = {"ner_" + k: v for k, v in metrics.items()}
        return metrics
