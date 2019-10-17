from typing import Dict, Optional, List, Any

import torch

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder
from allennlp.modules.matrix_attention.dot_product_matrix_attention import DotProductMatrixAttention
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, masked_softmax, weighted_sum
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure
from dygie.training.thresholding_f1_metric import BinaryThresholdF1


@Model.register("standard_coreference")
class StandardCoreference(Model):
    """
    Copied from Allennlp : Decomposable Attention
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        attend_feedforward: FeedForward,
        compare_feedforward: FeedForward,
        aggregate_feedforward: FeedForward,
        encoder: Optional[Seq2SeqEncoder] = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super(EntityLinker, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._attend_feedforward = TimeDistributed(attend_feedforward)
        self._matrix_attention = DotProductMatrixAttention()
        self._compare_feedforward = TimeDistributed(compare_feedforward)
        self._aggregate_feedforward = aggregate_feedforward
        self._encoder = encoder

        self._num_labels = vocab.get_vocab_size(namespace="labels")

        self._loss = torch.nn.CrossEntropyLoss()
        self._label_names = [None] * self._num_labels
        for k, v in vocab.get_token_to_index_vocabulary("labels").items():
            self._label_names[v] = k

        self._f1 = BinaryThresholdF1()

        initializer(self)

    def forward(
        self,  # type: ignore
        premise: Dict[str, torch.LongTensor],
        hypothesis: Dict[str, torch.LongTensor],
        pair_features: torch.Tensor = None,
        label: torch.IntTensor = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        premise : Dict[str, torch.LongTensor]
            From a ``TextField``
        hypothesis : Dict[str, torch.LongTensor]
            From a ``TextField``
        label : torch.IntTensor, optional, (default = None)
            From a ``LabelField``
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            Metadata containing the original tokenization of the premise and
            hypothesis with 'premise_tokens' and 'hypothesis_tokens' keys respectively.
        Returns
        -------
        An output dictionary consisting of:

        label_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing unnormalised log
            probabilities of the entailment label.
        label_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing probabilities of the
            entailment label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embedded_premise = self._text_field_embedder(premise)
        embedded_hypothesis = self._text_field_embedder(hypothesis)

        premise_mask = get_text_field_mask(premise).float()
        hypothesis_mask = get_text_field_mask(hypothesis).float()

        embedded_premise = self._encoder(embedded_premise, premise_mask)
        embedded_hypothesis = self._encoder(embedded_hypothesis, hypothesis_mask)

        projected_premise = self._attend_feedforward(embedded_premise)
        projected_hypothesis = self._attend_feedforward(embedded_hypothesis)
        # Shape: (batch_size, premise_length, hypothesis_length)
        similarity_matrix = self._matrix_attention(projected_premise, projected_hypothesis)

        # Shape: (batch_size, premise_length, hypothesis_length)
        p2h_attention = masked_softmax(similarity_matrix, hypothesis_mask)
        # Shape: (batch_size, premise_length, embedding_dim)
        attended_hypothesis = weighted_sum(embedded_hypothesis, p2h_attention)

        # Shape: (batch_size, hypothesis_length, premise_length)
        h2p_attention = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), premise_mask)
        # Shape: (batch_size, hypothesis_length, embedding_dim)
        attended_premise = weighted_sum(embedded_premise, h2p_attention)

        premise_compare_input = torch.cat([embedded_premise, attended_hypothesis], dim=-1)
        hypothesis_compare_input = torch.cat([embedded_hypothesis, attended_premise], dim=-1)

        compared_premise = self._compare_feedforward(premise_compare_input)
        compared_premise = compared_premise * premise_mask.unsqueeze(-1)
        # Shape: (batch_size, compare_dim)
        compared_premise = compared_premise.sum(dim=1)

        compared_hypothesis = self._compare_feedforward(hypothesis_compare_input)
        compared_hypothesis = compared_hypothesis * hypothesis_mask.unsqueeze(-1)
        # Shape: (batch_size, compare_dim)
        compared_hypothesis = compared_hypothesis.sum(dim=1)

        aggregate_input = torch.cat([compared_premise, compared_hypothesis, pair_features], dim=-1)
        label_logits = self._aggregate_feedforward(aggregate_input)

        aggregate_input = torch.cat([compared_hypothesis, compared_premise, pair_features], dim=-1)
        label_logits += self._aggregate_feedforward(aggregate_input)

        label_logits /= 2

        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

        output_dict = {"label_probs": label_probs[..., 1]}

        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            self._f1(label_probs[..., 1], label.long().view(-1))
            output_dict["loss"] = loss

        if metadata is not None:
            output_dict["metadata"] = metadata

        return output_dict

    def decode(self, output_dict: Dict[str, torch.Tensor]):
        output_dict["label_probs"] = list(output_dict["label_probs"].detach().cpu().numpy())
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self._f1.get_metric(reset)
        metrics = {k if not k.startswith("total") else ("_" + k): v for k, v in metrics.items()}
        return metrics
