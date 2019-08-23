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
from dygie.training.ner_metrics import NERMetrics, NERMentionMetrics

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class NERTagger(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        mention_feedforward: FeedForward,
        label_namespace: str = "ner_labels",
        balancing_strategy: str = None,
        decoding_type: str = "all_decode",
        decoding_metric: str = "entropy",
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super(NERTagger, self).__init__(vocab, regularizer)
        self._label_namespace = label_namespace
        # Number of classes determine the output dimension of the final layer
        self._n_labels = vocab.get_vocab_size(label_namespace)

        entity_labels = ["", "Material", "Metric", "Task", "Method"]
        self._entity_labels = entity_labels
        self._n_entity_labels = 1 + 4
        self._n_link_labels = 1 + 1

        self._label_map = vocab.get_token_to_index_vocabulary(label_namespace)

        self._label_to_entity_map = [0] * self._n_labels
        self._label_to_link_map = [0] * self._n_labels

        for k, v in self._label_map.items():
            if k != "":
                k = k.split("_")
                self._label_to_entity_map[v] = entity_labels.index(k[1])
                self._label_to_link_map[v] = 1 if k[2] == "True" else 0

        self.register_buffer("_label_to_entity_index", torch.Tensor(self._label_to_entity_map).long())
        self.register_buffer("_label_to_link_index", torch.Tensor(self._label_to_link_map).long())

        self.register_buffer("_class_weight", torch.Tensor([1.0] * self._n_labels))
        self.register_buffer("_sample_prob", torch.Tensor([1.0] * self._n_labels))
        self._registered_loss_modifiers = False

        # Null label is needed to keep track of when calculating the metrics
        null_label = vocab.get_token_index("", label_namespace)
        assert null_label == 0  # If not, the dummy class won't correspond to the null label.

        self._mention_feedforward = TimeDistributed(mention_feedforward)
        self._ner_scorer = TimeDistributed(torch.nn.Linear(mention_feedforward.get_output_dim(), self._n_labels - 1))

        self._ner_mention_metrics = NERMentionMetrics(self._entity_labels)
        self._ner_metrics = NERMetrics(list(vocab.get_token_to_index_vocabulary(label_namespace).keys()))

        self._decoding_metric = decoding_metric
        self._decoding_type = decoding_type

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
        ner_scores = self._append_dummy_scores(self._ner_scorer(span_feedforward))
        predicted_ner = self.predict_ner_from_combined(ner_scores, span_mask)

        ner_probs = F.softmax(ner_scores, dim=-1)
        ner_linked_probs = ner_probs.new_zeros((ner_probs.size(0), ner_probs.size(1), self._n_link_labels))
        ner_linked_probs.index_add_(-1, self._label_to_link_index, ner_probs)

        ner_entity_probs = ner_probs.new_zeros((ner_probs.size(0), ner_probs.size(1), self._n_entity_labels))
        ner_entity_probs.index_add_(-1, self._label_to_entity_index, ner_probs)

        output_dict = {
            "spans" : spans,
            "span_mask" : span_mask,
            "ner_entity_scores": ner_entity_probs,
            "predicted_ner": predicted_ner,
            "ner_linked_scores": ner_linked_probs[:, :, 1].unsqueeze(-1),
        }

        if ner_labels is not None:
            ner_labels_str = self.map_combined_prediction_to_str(ner_labels.view(-1), span_mask)
            self._ner_metrics(predicted_ner, ner_labels_str)

            predicted_dict = self.all_decode(output_dict)
            true_dict = self.decode_gold_label(ner_labels, spans, span_mask)
            self._ner_mention_metrics(predicted_dict, true_dict)

            loss = self._compute_loss_for_scores(ner_scores, ner_labels, span_mask, metadata)

            output_dict["loss"] = loss

        if metadata is not None:
            output_dict["metadata"] = metadata

        return output_dict

    def _compute_loss_for_scores(self, ner_scores, ner_labels, span_mask, metadata):
        mask_flat = span_mask.view(-1).byte()

        ner_scores_flat = ner_scores.view(-1, ner_scores.size(-1))[mask_flat]
        ner_labels_flat = ner_labels.view(-1)[mask_flat]

        if not self._registered_loss_modifiers:
            sample_prob = metadata[0]["ner_labels_sample_prob"]
            for k, v in sample_prob.items():
                self._sample_prob[self._label_map[k]] = v

            class_weight = metadata[0]["ner_labels_class_weight"]
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

    @staticmethod
    def _append_dummy_scores(ner_scores):
        dummy_scores = ner_scores.new_zeros(ner_scores.size(0), ner_scores.size(1), 1)
        ner_scores = torch.cat((dummy_scores, ner_scores), -1)
        return ner_scores

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

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):
        return getattr(self, self._decoding_type)(output_dict)

    def all_decode(self, output_dict: Dict[str, torch.Tensor]):
        predicted_batch = torch.argmax(output_dict["ner_entity_scores"], dim=-1) #(B, Ns)
        spans_batch = output_dict["spans"].detach().cpu() #(B, Ns, 2)
        span_mask_batch = output_dict["span_mask"].detach().cpu().byte() #(B, Ns)

        res_dict = []
        for spans, spans_mask, spans_predicted_ner in zip(
            spans_batch, span_mask_batch, predicted_batch
        ):
            entry_dict = {}
            for span, ner_label in zip(spans[spans_mask], spans_predicted_ner[spans_mask]):
                the_span = (span[0].item(), span[1].item() + 1)
                the_label = self._entity_labels[ner_label.item()]
                entry_dict[the_span] = the_label

            res_dict.append(entry_dict)

        output_dict['decoded_ner'] = res_dict
        return output_dict

    def dp_decode(self, output_dict: Dict[str, torch.Tensor]):
        dp_scores_batch = getattr(self, self._decoding_metric)(output_dict["ner_entity_scores"]) #(B, Ns,)
        predicted_batch = torch.argmax(output_dict["ner_entity_scores"], dim=-1) #(B, Ns)
        spans_batch = output_dict["spans"].detach().cpu() #(B, Ns, 2)
        span_mask_batch = output_dict["span_mask"].detach().cpu().byte() #(B, Ns)

        res_dict = []
        for spans, spans_mask, spans_predicted_ner, spans_dp_scores in zip(
            spans_batch, span_mask_batch, predicted_batch, dp_scores_batch
        ):
            entry_dict = {}
            spans, spans_predicted_ner, spans_dp_scores = (
                spans[spans_mask],
                spans_predicted_ner[spans_mask],
                spans_dp_scores[spans_mask],
            )
            length, max_span_length = 0, 0
            for span, ner_label, dp_score in zip(spans, spans_predicted_ner, spans_dp_scores):
                the_span = (span[0].item(), span[1].item())
                length = max(length, the_span[1] + 1)
                max_span_length = max(max_span_length, the_span[1] - the_span[0] + 1)
                the_label = self._entity_labels[ner_label.item()]
                entry_dict[the_span] = (the_label, dp_score.item())

            entry_dict = self.run_dynamic_programming(entry_dict, length, max_span_length)
            res_dict.append(entry_dict)

        return res_dict

    @staticmethod
    def run_dynamic_programming(entry_dict, doc_length, max_span_length):
        base_score = float("-inf")
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
                if curr_score > doc_length_best_score[i]:
                    doc_length_best_score[i] = curr_score
                    doc_length_position[i] = last_index

        decoded_entry_dict = {}
        backlink = doc_length - 1
        while backlink >= 0:
            best_last_index = doc_length_position[backlink]
            span_start = best_last_index + 1
            span_end = backlink
            decoded_entry_dict[(span_start, span_end)] = entry_dict[(span_start, span_end)][0]
            backlink = best_last_index

        return decoded_entry_dict

    def entropy(self, scores: torch.Tensor):
        h = scores.max(-1)[0]
        return h

    def decode_gold_label(self, gold_labels_batched, spans_batched, spans_mask_batched) :
        mask_batched = (spans_mask_batched.long() * (gold_labels_batched > 0).long()).byte()
        res_dict = []
        for spans, spans_mask, gold_labels in zip(spans_batched, mask_batched, gold_labels_batched) :
            spans = spans[spans_mask]
            gold_labels = gold_labels[spans_mask]
            true_labels_dict = {}
            for span, gold_label in zip(spans, gold_labels) :
                true_labels_dict[(span[0].item(), span[1].item())] = self._entity_labels[self._label_to_entity_map[gold_label.item()]]
            res_dict.append(true_labels_dict)
        return res_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self._ner_metrics.get_metric(reset)
        metrics = {"ner_" + k: v for k, v in metrics.items()}

        mention_metrics = self._ner_mention_metrics.get_metric(reset)
        metrics.update({'mention_ner_' + k:v for k, v in mention_metrics.items()})

        return metrics

