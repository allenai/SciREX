import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.modules import FeedForward
from allennlp.modules import TimeDistributed
from dygie.models.prescored_pruner import PrescoredPruner
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from dygie.training.relation_metrics import MentionRecall, CorefScores

from dygie.models import shared

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class CorefResolver(Model):
    """
    TODO(dwadden) document correctly.

    Parameters
    ----------
    mention_feedforward : ``FeedForward``
        This feedforward network is applied to the span representations which is then scored
        by a linear layer.
    antecedent_feedforward: ``FeedForward``
        This feedforward network is applied to pairs of span representation, along with any
        pairwise features, which is then scored by a linear layer.
    feature_size: ``int``
        The embedding size for all the embedded features, such as distances or span widths.
    spans_per_word: float, required.
        A multiplier between zero and one which controls what percentage of candidate mention
        spans we retain with respect to the number of words in the document.
    max_antecedents: int, required.
        For each mention which survives the pruning stage, we consider this many antecedents.
    lexical_dropout: ``int``
        The probability of dropping out dimensions of the embedded text.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        antecedent_feedforward: FeedForward,
        spans_per_word: float,
        max_antecedents: int,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super(CorefResolver, self).__init__(vocab, regularizer)

        self._antecedent_feedforward = TimeDistributed(antecedent_feedforward)
        self._mention_pruner = PrescoredPruner()
        self._antecedent_scorer = TimeDistributed(torch.nn.Linear(antecedent_feedforward.get_output_dim(), 1))

        self._score_mixer = torch.nn.Linear(4, 1, bias=False)

        self._spans_per_word = spans_per_word
        self._max_antecedents = max_antecedents

        self._mention_recall = MentionRecall()
        self._coref_scores_1 = CorefScores()
        self._coref_scores_0 = CorefScores()

        initializer(self)

    def compute_representations(
        self,  # type: ignore
        spans_batched: torch.IntTensor,
        span_mask_batched,
        span_embeddings_batched,
        span_scores_batched,
        span_ner_scores_dist_batched,
        sentence_lengths,
        coref_labels_batched: torch.IntTensor = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Run the forward pass for a single document.

        Important: This function assumes that sentences are going to be passed in in sorted order,
        from the same document.
        """

        span_ix = span_mask_batched.view(-1).nonzero().squeeze()  # Indices of the spans to keep.
        spans = self._flatten_spans(span_ix, spans_batched, sentence_lengths)

        # Flatten the span embeddings and keep the good ones.
        span_embeddings = self._flatten_span_info(span_embeddings_batched, span_ix)
        span_scores = self._flatten_span_info(span_scores_batched, span_ix)
        span_ner_scores_dist = self._flatten_span_info(span_ner_scores_dist_batched, span_ix)
        coref_labels = self._flatten_span_info(coref_labels_batched, span_ix)

        document_length = sentence_lengths.sum().item()
        num_spans = spans.size(1)

        # Prune based on mention scores. Make sure we keep at least 1.
        num_spans_to_keep = min(max(2, int(math.ceil(self._spans_per_word * document_length))), 100)

        # Since there's only one minibatch, there aren't any masked spans for us. The span mask is
        # always 1.
        span_mask = torch.ones(num_spans, device=spans_batched.device).unsqueeze(0)

        # Shape: (1, num_spans_to_keep, E), (1, num_spans_to_keep), (1, num_spans_to_keep), (1, num_spans_to_keep, 1)
        (top_span_embeddings, top_span_mask, top_span_indices, top_span_mention_scores) = self._mention_pruner(
            span_embeddings, span_mask, num_spans_to_keep, span_scores
        )
        top_span_mask = top_span_mask.unsqueeze(-1)
        # Shape: (batch_size * num_spans_to_keep)
        flat_top_span_indices = util.flatten_and_batch_shift_indices(top_span_indices, num_spans)

        top_span_ner_scores_dist = util.batched_index_select(
                span_ner_scores_dist, top_span_indices, flat_top_span_indices
            )

        # Compute final predictions for which spans to consider as mentions.
        # Shape: (batch_size, num_spans_to_keep, 2)
        top_spans = util.batched_index_select(spans, top_span_indices, flat_top_span_indices)

        # Compute indices for antecedent spans to consider.
        max_antecedents = num_spans_to_keep

        # Shapes:
        # (num_spans_to_keep, max_antecedents),
        # (1, max_antecedents),
        # (1, num_spans_to_keep, max_antecedents)
        valid_antecedent_indices, valid_antecedent_offsets, valid_antecedent_log_mask = self._generate_valid_antecedents(
            num_spans_to_keep, max_antecedents, util.get_device_of(span_embeddings)
        )

        coreference_scores = self.get_coref_scores(
            top_span_embeddings,
            top_span_mention_scores,
            top_span_ner_scores_dist,
            valid_antecedent_indices,
            valid_antecedent_offsets,
            valid_antecedent_log_mask,
        )

        output_dict = {
            "top_spans": top_spans,  # (1, num_spans_to_keep, 2)
            "antecedent_indices": valid_antecedent_indices,  # (num_spans_to_keep, max_antecendents)
            "valid_antecedent_log_mask": valid_antecedent_log_mask,  # (1, num_spans_to_keep, max_antecedents)
            "valid_antecedent_offsets": valid_antecedent_offsets,  # (1, max_antecedents)
            "top_span_indices": top_span_indices,  # (1, num_spans_to_keep)
            "top_span_mask": top_span_mask,  # (1, num_spans_to_keep, 1)
            "top_span_embeddings": top_span_embeddings,  # (1, num_spans_to_keep, E)
            "flat_top_span_indices": flat_top_span_indices,  # (num_spans_to_keep,)
            "coref_labels": coref_labels,  # (1, num_spans, n_LE)
            "coreference_scores": coreference_scores,  # (1, num_spans, max_antecedent + 1)
            "sentence_lengths": sentence_lengths,  # (B,)
            "span_ix": span_ix,  # (num_spans,)
            "metadata": metadata,
        }

        return output_dict

    def get_coref_scores(
        self,
        top_span_embeddings,
        top_span_mention_scores,
        top_span_ner_scores_dist,
        valid_antecedent_indices,
        valid_antecedent_offsets,
        valid_antecedent_log_mask,
    ):
        candidate_antecedent_embeddings = util.flattened_index_select(top_span_embeddings, valid_antecedent_indices)
        # Shape: (batch_size, num_spans_to_keep, max_antecedents)
        candidate_antecedent_mention_scores = util.flattened_index_select(
            top_span_mention_scores, valid_antecedent_indices
        ).squeeze(-1)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents, ner_labels)
        candidate_antecedent_ner_scores_dist = util.flattened_index_select(
            top_span_ner_scores_dist, valid_antecedent_indices
        )
        # Compute antecedent scores.
        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        span_pair_embeddings = self._compute_span_pair_embeddings(
            top_span_embeddings, candidate_antecedent_embeddings, valid_antecedent_offsets
        )
        # Shape: (batch_size, num_spans_to_keep, 1 + max_antecedents)
        coreference_scores = self._compute_coreference_scores(
            span_pair_embeddings,
            top_span_mention_scores,
            top_span_ner_scores_dist,
            candidate_antecedent_mention_scores,
            candidate_antecedent_ner_scores_dist,
            valid_antecedent_log_mask,
        )
        return coreference_scores

    def predict_labels(self, output_dict):
        coref_labels = output_dict["coref_labels"]  # (1, num_spans, n_LE)
        coreference_scores = output_dict["coreference_scores"]  # (1, num_spans_to_keep, max_antecedent)

        predicted_antecedents = (coreference_scores > 0.5).long()
        output_dict["predicted_antecedents"] = predicted_antecedents

        top_span_indices = output_dict["top_span_indices"]
        flat_top_span_indices = output_dict["flat_top_span_indices"]
        valid_antecedent_indices = output_dict["antecedent_indices"]
        valid_antecedent_log_mask = output_dict["valid_antecedent_log_mask"]

        if coref_labels is not None:
            # breakpoint()
            # Find the gold labels for the spans which we kept.
            pruned_gold_labels = util.batched_index_select(
                coref_labels, top_span_indices, flat_top_span_indices
            )  # (1, num_spans_to_keep, n_LE)

            # (1, num_spans_to_keep, max_antecedents, n_LE)
            antecedent_labels = util.flattened_index_select(pruned_gold_labels, valid_antecedent_indices)
            # There's an integer wrap-around happening here. It occurs in the original code.
            antecedent_labels *= valid_antecedent_log_mask.exp().long().unsqueeze(-1)

            # Compute labels.
            # Shape: (1, num_spans_to_keep, max_antecedents)
            gold_antecedent_labels, linked_indicator = self._compute_antecedent_gold_labels(
                pruned_gold_labels, antecedent_labels
            )
            # Now, compute the loss using the negative marginal log-likelihood.
            output_dict["loss"] = F.binary_cross_entropy(
                coreference_scores,
                gold_antecedent_labels,
                weight=linked_indicator + 2 * gold_antecedent_labels,
            )

            true_spans_with_coref = (coref_labels.sum(-1) > 0).long().squeeze(0)
            predicted_spans_with_coref = torch.zeros(
                *true_spans_with_coref.size(), device=true_spans_with_coref.device
            ).long()
            predicted_spans_with_coref[top_span_indices] = 1

            self._mention_recall(predicted_spans_with_coref, true_spans_with_coref)
            self._coref_scores_1(predicted_antecedents.squeeze(0).long(), gold_antecedent_labels.squeeze(0).long())
            self._coref_scores_0(
                1 - predicted_antecedents.squeeze(0).long(), 1 - gold_antecedent_labels.squeeze(0).long()
            )

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):
        """
        Converts the list of spans and predicted antecedent indices into clusters
        of spans for each element in the batch.

        Parameters
        ----------
        output_dict : ``Dict[str, torch.Tensor]``, required.
            The result of calling :func:`forward` on an instance or batch of instances.

        Returns
        -------
        The same output dictionary, but with an additional ``clusters`` key:

        clusters : ``List[List[List[Tuple[int, int]]]]``
            A nested list, representing, for each instance in the batch, the list of clusters,
            which are in turn comprised of a list of (start, end) inclusive spans into the
            original document.
        """

        # A tensor of shape (batch_size, num_spans_to_keep, 2), representing
        # the start and end indices of each span.
        batch_top_spans = output_dict["top_spans"].detach().cpu()

        # A tensor of shape (batch_size, num_spans_to_keep) representing, for each span,
        # the index into ``antecedent_indices`` which specifies the antecedent span. Additionally,
        # the index can be -1, specifying that the span has no predicted antecedent.
        batch_predicted_antecedents = output_dict["predicted_antecedents"].detach().cpu()

        # A tensor of shape (num_spans_to_keep, max_antecedents), representing the indices
        # of the predicted antecedents with respect to the 2nd dimension of ``batch_top_spans``
        # for each antecedent we considered.
        antecedent_indices = output_dict["antecedent_indices"].detach().cpu()
        batch_clusters: List[List[List[Tuple[int, int]]]] = []

        # Calling zip() on two tensors results in an iterator over their
        # first dimension. This is iterating over instances in the batch.
        for top_spans, predicted_antecedents in zip(batch_top_spans, batch_predicted_antecedents):
            spans_to_cluster_ids: Dict[Tuple[int, int], int] = {}
            clusters: List[List[Tuple[int, int]]] = []

            for i, (span, predicted_antecedent) in enumerate(zip(top_spans, predicted_antecedents)):
                if predicted_antecedent < 0:
                    # We don't care about spans which are
                    # not co-referent with anything.
                    continue

                # Find the right cluster to update with this span.
                predicted_index = antecedent_indices[i, predicted_antecedent]

                antecedent_span = (top_spans[predicted_index, 0].item(), top_spans[predicted_index, 1].item())

                # Check if we've seen the span before.
                if antecedent_span in spans_to_cluster_ids:
                    predicted_cluster_id: int = spans_to_cluster_ids[antecedent_span]
                else:
                    # We start a new cluster.
                    predicted_cluster_id = len(clusters)
                    # Append a new cluster containing only this span.
                    clusters.append([antecedent_span])
                    # Record the new id of this span.
                    spans_to_cluster_ids[antecedent_span] = predicted_cluster_id

                # Now add the span we are currently considering.
                span_start, span_end = span[0].item(), span[1].item()
                clusters[predicted_cluster_id].append((span_start, span_end))
                spans_to_cluster_ids[(span_start, span_end)] = predicted_cluster_id
            batch_clusters.append(clusters)

        output_dict["clusters"] = batch_clusters
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        mention_recall = self._mention_recall.get_metric(reset)
        coref_precision_1, coref_recall_1, coref_f1_1 = self._coref_scores_1.get_metric(reset)
        coref_precision_0, coref_recall_0, coref_f1_0 = self._coref_scores_0.get_metric(reset)

        return {
            "coref_precision_1": coref_precision_1,
            "coref_recall_1": coref_recall_1,
            "coref_f1_1": coref_f1_1,
            "coref_precision_0": coref_precision_0,
            "coref_recall_0": coref_recall_0,
            "coref_f1_0": coref_f1_0,
            "coref_mention_recall": mention_recall
        }

    @staticmethod
    def _generate_valid_antecedents(
        num_spans_to_keep: int, max_antecedents: int, device: int
    ) -> Tuple[torch.IntTensor, torch.IntTensor, torch.FloatTensor]:
        """
        This method generates possible antecedents per span which survived the pruning
        stage. This procedure is `generic across the batch`. The reason this is the case is
        that each span in a batch can be coreferent with any previous span, but here we
        are computing the possible `indices` of these spans. So, regardless of the batch,
        the 1st span _cannot_ have any antecedents, because there are none to select from.
        Similarly, each element can only predict previous spans, so this returns a matrix
        of shape (num_spans_to_keep, max_antecedents), where the (i,j)-th index is equal to
        (i - 1) - j if j <= i, or zero otherwise.

        Parameters
        ----------
        num_spans_to_keep : ``int``, required.
            The number of spans that were kept while pruning.
        max_antecedents : ``int``, required.
            The maximum number of antecedent spans to consider for every span.
        device: ``int``, required.
            The CUDA device to use.

        Returns
        -------
        valid_antecedent_indices : ``torch.IntTensor``
            The indices of every antecedent to consider with respect to the top k spans.
            Has shape ``(num_spans_to_keep, max_antecedents)``.
        valid_antecedent_offsets : ``torch.IntTensor``
            The distance between the span and each of its antecedents in terms of the number
            of considered spans (i.e not the word distance between the spans).
            Has shape ``(1, max_antecedents)``.
        valid_antecedent_log_mask : ``torch.FloatTensor``
            The logged mask representing whether each antecedent span is valid. Required since
            different spans have different numbers of valid antecedents. For example, the first
            span in the document should have no valid antecedents.
            Has shape ``(1, num_spans_to_keep, max_antecedents)``.
        """
        # Shape: (num_spans_to_keep, 1)
        target_indices = util.get_range_vector(num_spans_to_keep, device).unsqueeze(1)

        # Shape: (1, max_antecedents)
        valid_antecedent_offsets = (util.get_range_vector(max_antecedents, device) + 1).unsqueeze(0)

        # This is a broadcasted subtraction.
        # Shape: (num_spans_to_keep, max_antecedents)
        raw_antecedent_indices = target_indices - valid_antecedent_offsets

        # Shape: (1, num_spans_to_keep, max_antecedents)
        valid_antecedent_log_mask = (raw_antecedent_indices >= 0).float().unsqueeze(0).log()

        # Shape: (num_spans_to_keep, max_antecedents)
        valid_antecedent_indices = F.relu(raw_antecedent_indices.float()).long()
        return valid_antecedent_indices, valid_antecedent_offsets, valid_antecedent_log_mask

    def _compute_span_pair_embeddings(
        self,
        top_span_embeddings: torch.FloatTensor,
        antecedent_embeddings: torch.FloatTensor,
        antecedent_offsets: torch.FloatTensor,
    ):
        """
        Computes an embedding representation of pairs of spans for the pairwise scoring function
        to consider. This includes both the original span representations, the element-wise
        similarity of the span representations, and an embedding representation of the distance
        between the two spans.

        Parameters
        ----------
        top_span_embeddings : ``torch.FloatTensor``, required.
            Embedding representations of the top spans. Has shape
            (batch_size, num_spans_to_keep, embedding_size).
        antecedent_embeddings : ``torch.FloatTensor``, required.
            Embedding representations of the antecedent spans we are considering
            for each top span. Has shape
            (batch_size, num_spans_to_keep, max_antecedents, embedding_size).
        antecedent_offsets : ``torch.IntTensor``, required.
            The offsets between each top span and its antecedent spans in terms
            of spans we are considering. Has shape (1, max_antecedents).

        Returns
        -------
        span_pair_embeddings : ``torch.FloatTensor``
            Embedding representation of the pair of spans to consider. Has shape
            (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        """
        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        target_embeddings = top_span_embeddings.unsqueeze(2).expand_as(antecedent_embeddings)

        # Shape: (1, max_antecedents, embedding_size)
        antecedent_distance_embeddings = self._distance_embedding(
            util.bucket_values(antecedent_offsets, num_total_buckets=self._num_distance_buckets)
        )

        # Shape: (1, 1, max_antecedents, embedding_size)
        antecedent_distance_embeddings = antecedent_distance_embeddings.unsqueeze(0)

        expanded_distance_embeddings_shape = (
            antecedent_embeddings.size(0),
            antecedent_embeddings.size(1),
            antecedent_embeddings.size(2),
            antecedent_distance_embeddings.size(-1),
        )
        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        antecedent_distance_embeddings = antecedent_distance_embeddings.expand(*expanded_distance_embeddings_shape)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        span_pair_embeddings = torch.cat(
            [
                target_embeddings,
                antecedent_embeddings,
                antecedent_embeddings * target_embeddings,
                antecedent_distance_embeddings,
            ],
            -1,
        )
        return span_pair_embeddings

    @staticmethod
    def _compute_antecedent_gold_labels(top_coref_labels: torch.IntTensor, antecedent_labels: torch.IntTensor):
        """
        Generates a binary indicator for every pair of spans. This label is one if and
        only if the pair of spans belong to the same cluster. The labels are augmented
        with a dummy antecedent at the zeroth position, which represents the prediction
        that a span does not have any antecedent.

        Parameters
        ----------
        top_coref_labels : ``torch.IntTensor``, required.
            The cluster id label for every span. The id is arbitrary,
            as we just care about the clustering. Has shape (batch_size, num_spans_to_keep, n_LE).
        antecedent_labels : ``torch.IntTensor``, required.
            The cluster id label for every antecedent span. The id is arbitrary,
            as we just care about the clustering. Has shape
            (batch_size, num_spans_to_keep, max_antecedents, n_LE).

        Returns
        -------
        pairwise_labels_with_dummy_label : ``torch.FloatTensor``
            A binary tensor representing whether a given pair of spans belong to
            the same cluster in the gold clustering.
            Has shape (batch_size, num_spans_to_keep, max_antecedents + 1).

        """
        # Shape: (batch_size, num_spans_to_keep, max_antecedents)
        target_labels = top_coref_labels.unsqueeze(2)
        same_cluster_indicator = (target_labels * antecedent_labels).sum(-1).byte().float()
        linked_indicator = (target_labels + antecedent_labels).sum(-1).byte().float()
        pairwise_labels = same_cluster_indicator  # (1, num_spans_to_keep, max_antecedents)
        return pairwise_labels, linked_indicator

    def _compute_coreference_scores(
        self,
        pairwise_embeddings: torch.FloatTensor,
        top_span_mention_scores: torch.FloatTensor,
        top_span_ner_scores_dist: torch.FloatTensor,
        antecedent_mention_scores: torch.FloatTensor,
        candidate_antecedent_ner_scores_dist: torch.FloatTensor,
        antecedent_log_mask: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Computes scores for every pair of spans. Additionally, a dummy label is included,
        representing the decision that the span is not coreferent with anything. For the dummy
        label, the score is always zero. For the true antecedent spans, the score consists of
        the pairwise antecedent score and the unary mention scores for the span and its
        antecedent. The factoring allows the model to blame many of the absent links on bad
        spans, enabling the pruning strategy used in the forward pass.

        Parameters
        ----------
        pairwise_embeddings: ``torch.FloatTensor``, required.
            Embedding representations of pairs of spans. Has shape
            (batch_size, num_spans_to_keep, max_antecedents, encoding_dim)
        top_span_mention_scores: ``torch.FloatTensor``, required.
            Mention scores for every span. Has shape
            (batch_size, num_spans_to_keep, max_antecedents).
        top_span_ner_scores_dist: ``torch.FloatTensor``, required.
            (batch_size, num_spans_to_keep, ner_labels)
        antecedent_mention_scores: ``torch.FloatTensor``, required.
            Mention scores for every antecedent. Has shape
            (batch_size, num_spans_to_keep, max_antecedents).
        antecedent_log_mask: ``torch.FloatTensor``, required.
            The log of the mask for valid antecedents.

        Returns
        -------
        coreference_scores: ``torch.FloatTensor``
            A tensor of shape (batch_size, num_spans_to_keep, max_antecedents + 1),
            representing the unormalised score for each (span, antecedent) pair
            we considered.

        """
        # Shape: (batch_size, num_spans_to_keep, max_antecedents)
        antecedent_scores = self._antecedent_scorer(self._antecedent_feedforward(pairwise_embeddings))
        top_span_ner_scores_dist = top_span_ner_scores_dist.unsqueeze(2) #(B, NS, MA, l)
        kl_score = self.safe_kl_div(top_span_ner_scores_dist, candidate_antecedent_ner_scores_dist)
        kl_score += self.safe_kl_div(candidate_antecedent_ner_scores_dist, top_span_ner_scores_dist)
        kl_score = torch.exp(-kl_score)

        antecedent_scores = torch.cat(
            [
                top_span_mention_scores.unsqueeze(-1).expand_as(antecedent_scores),
                antecedent_mention_scores.unsqueeze(-1),
                antecedent_scores,
                kl_score.unsqueeze(-1)
            ],
            dim=-1,
        )
        antecedent_scores = self._score_mixer(antecedent_scores).squeeze(-1)
        antecedent_scores += antecedent_log_mask

        antecedent_scores = torch.sigmoid(antecedent_scores) * kl_score

        return antecedent_scores

    @staticmethod
    def safe_kl_div(p, q, dim=-1) :
        return (p * torch.log(p/(q + 1e-10))).sum(dim)

    @staticmethod
    def _flatten_spans(span_ix, spans_batched, sentence_lengths):
        """
        Spans are input with each minibatch as a sentence. For coref, it's easier to flatten them out
        and consider all sentences together as a document.
        """

        # Change the span offsets to document-level, flatten, and keep good ones.
        sentence_offset = shared.cumsum_shifted(sentence_lengths).unsqueeze(1).unsqueeze(2)
        assert len(sentence_offset.shape) == 3 and sentence_offset.size(1) == 1 and sentence_offset.size(2) == 1
        spans_offset = spans_batched + sentence_offset
        spans_flat = spans_offset.view(-1, 2)
        spans_flat = spans_flat[span_ix].unsqueeze(0)

        # (1, Total Spans, *)
        return spans_flat

    @staticmethod
    def _flatten_span_info(span_info_batched, span_ix):
        feature_size = span_info_batched.size(-1)
        emb_flat = span_info_batched.view(-1, feature_size)
        span_info_flat = emb_flat[span_ix].unsqueeze(0)
        return span_info_flat

