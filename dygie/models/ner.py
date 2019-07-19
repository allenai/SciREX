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

    def __init__(self,
                 vocab: Vocabulary,
                 mention_feedforward: FeedForward,
                 decoding_type:str = 'dp_decode',
                 decoding_metric:str = 'entropy',
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(NERTagger, self).__init__(vocab, regularizer)

        # Number of classes determine the output dimension of the final layer
        self._n_labels = vocab.get_vocab_size('ner_labels')

        # TODO(dwadden) think of a better way to enforce this.
        # Null label is needed to keep track of when calculating the metrics
        null_label = vocab.get_token_index("", "ner_labels")
        assert null_label == 0  # If not, the dummy class won't correspond to the null label.

        self._ner_scorer = torch.nn.Sequential(
            TimeDistributed(mention_feedforward),
            TimeDistributed(torch.nn.Linear(
                mention_feedforward.get_output_dim(),
                self._n_labels - 1)))

        self._ner_metrics = NERMetrics(self._n_labels, null_label)

        self._loss = torch.nn.CrossEntropyLoss(reduction="sum")

        self._decoding_metric = decoding_metric
        self._decoding_type = decoding_type

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                spans: torch.IntTensor,
                span_mask: torch.IntTensor,
                span_embeddings: torch.IntTensor,
                sentence_lengths: torch.Tensor,
                ner_labels: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        """
        TODO(dwadden) Write documentation.
        """

        # Shape: (Batch size, Number of Spans, Span Embedding Size)
        # span_embeddings

        #Shape: (Batch_size, Number of spans, n_labels - 1)
        ner_scores = self._ner_scorer(span_embeddings)
        dummy_scores = ner_scores.new_zeros(ner_scores.size(0), ner_scores.size(1), 1)
        ner_scores = torch.cat((dummy_scores, ner_scores), -1)

        _, predicted_ner = ner_scores.max(2)

        output_dict = {"spans": spans,
                       "span_mask": span_mask,
                       "ner_scores": ner_scores,
                       "predicted_ner": predicted_ner}

        if ner_labels is not None:
            self._ner_metrics(predicted_ner, ner_labels, span_mask)
            ner_scores_flat = ner_scores.view(-1, self._n_labels)
            ner_labels_flat = ner_labels.view(-1)
            mask_flat = span_mask.view(-1).byte()

            loss = self._loss(ner_scores_flat[mask_flat], ner_labels_flat[mask_flat])
            output_dict["loss"] = loss

        if metadata is not None:
            output_dict["document"] = [x["sentence"] for x in metadata]

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) :
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
        dp_scores = getattr(self, self._decoding_metric)(output_dict['ner_scores'])
        predicted_ner_batch = output_dict["predicted_ner"].detach().cpu()
        spans_batch = output_dict["spans"].detach().cpu()
        span_mask_batch = output_dict["span_mask"].detach().cpu().byte()

        res_list = []
        res_dict = []
        for spans, spans_mask, spans_predicted_ner, spans_dp_scores in zip(spans_batch, span_mask_batch, predicted_ner_batch, dp_scores) :
            entry_dict = {}
            spans, spans_predicted_ner, spans_dp_scores = spans[spans_mask], spans_predicted_ner[spans_mask], spans_dp_scores[spans_mask]
            for span, ner_label, dp_score in zip(spans, spans_predicted_ner, spans_dp_scores) :
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

    def run_dynamic_programming(self, entry_dict, doc_length, max_span_length) :
        base_score = float('inf')
        doc_length_position = [-1] * doc_length
        doc_length_best_score = [base_score] * doc_length

        for i in range(doc_length) :
            for j in range(1, max_span_length + 1) :
                span_end = i
                span_start = span_end - j + 1
                if span_start < 0 : continue
                last_index = span_start - 1
                curr_span_score = entry_dict[(span_start, span_end)][1]
                last_index_score = doc_length_best_score[last_index] if last_index >= 0 else 0
                curr_score = last_index_score + curr_span_score
                if curr_score < doc_length_best_score[i] :
                    doc_length_best_score[i] = curr_score
                    doc_length_position[i] = last_index

        decoded_entry_dict = {}
        backlink = doc_length - 1
        while backlink >= 0 :
            best_last_index = doc_length_position[backlink]
            span_start = best_last_index + 1
            span_end = backlink
            if entry_dict[(span_start, span_end)][0] > 0 :
                decoded_entry_dict[(span_start, span_end)] = entry_dict[(span_start, span_end)][0]
            backlink = best_last_index

        return decoded_entry_dict

    def entropy(self, scores: torch.Tensor) :
        p, log_p = F.softmax(scores, dim=-1), F.log_softmax(scores, dim=-1)
        h = -(p * log_p).sum(-1)
        return h

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        ner_precision, ner_recall, ner_f1 = self._ner_metrics.get_metric(reset)
        return {"ner_precision": ner_precision,
                "ner_recall": ner_recall,
                "ner_f1": ner_f1}
