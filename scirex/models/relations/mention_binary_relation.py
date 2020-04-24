import logging
from typing import Any, Dict, List, Optional
import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import TimeDistributed
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from scirex.metrics.thresholding_f1_metric import BinaryThresholdF1

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class RelationExtractor(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        antecedent_feedforward: FeedForward,
        relation_coverage: int = 0,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super(RelationExtractor, self).__init__(vocab, regularizer)

        self._antecedent_feedforward = TimeDistributed(antecedent_feedforward)
        self._antecedent_scorer = TimeDistributed(torch.nn.Linear(antecedent_feedforward.get_output_dim(), 1))

        self._coref_scores = BinaryThresholdF1()

        self._relation_coverage = relation_coverage

        initializer(self)

    def compute_representations(
        self,  # type: ignore
        spans_batched: torch.IntTensor,
        span_mask_batched,
        span_embeddings_batched,
        coref_labels_batched: torch.IntTensor = None,
        relation_index_batched: torch.IntTensor = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Run the forward pass for a single document.

        Important: This function assumes that sentences are going to be passed in in sorted order,
        from the same document.
        """

        start_pos_in_doc = torch.LongTensor([x["start_pos_in_doc"] for x in metadata]).to(spans_batched.device)
        section_offset = start_pos_in_doc.unsqueeze(1).unsqueeze(2)
        assert len(section_offset.shape) == 3 and section_offset.size(1) == 1 and section_offset.size(2) == 1
        spans_offset = spans_batched + section_offset

        span_ix = span_mask_batched.view(-1).nonzero().squeeze(1).long()  # Indices of the spans to keep.
        spans, spans_para_idx = self._flatten_spans(span_ix, spans_offset)
        span_embeddings = self._flatten_span_info(span_embeddings_batched, span_ix)

        if span_embeddings.nelement() == 0 :
            breakpoint()

        coref_labels, relation_labels = None, None
        if coref_labels_batched is not None and relation_index_batched is not None:
            coref_labels = self._flatten_span_info(coref_labels_batched, span_ix)
            relation_index = relation_index_batched[0].unsqueeze(0).transpose(1, 2)
            assert len(relation_index.shape) == 3 and relation_index.shape[1] == coref_labels.shape[2]
            relation_labels = torch.bmm(coref_labels.float(), relation_index.float()).clamp(0, 1).long()
            assert len(relation_labels.shape) == len(coref_labels.shape)

        num_spans = spans.size(1)
        span_mask = torch.ones(num_spans, device=spans_batched.device).unsqueeze(0)
        relation_scores = self.get_relation_scores(spans, span_embeddings)

        output_dict = {
            "spans": spans,  # (1, num_spans_to_keep, 2)
            "span_mask": span_mask,  # (1, num_spans_to_keep, 1)
            "relation_labels": relation_labels,  # (1, num_spans, n_LE)
            "coref_labels": coref_labels,
            "relation_scores": relation_scores,  # (1, num_spans, max_antecedent + 1)
            "span_ix": span_ix,  # (num_spans,)
            "metadata": metadata,
            "spans_para_idx": spans_para_idx,
            "batch_size" : span_embeddings_batched.shape[0]
        }

        output_dict = self.predict_labels(output_dict)

        return output_dict

    def get_relation_scores(self, spans, top_span_embeddings):
        # (B, NS, NS, E)
        span_pair_embeddings = self._compute_span_pair_embeddings(spans, top_span_embeddings)
        relation_scores = torch.sigmoid(self._antecedent_scorer(self._antecedent_feedforward(span_pair_embeddings)))
        return relation_scores

    def predict_labels(self, output_dict):
        relation_labels = output_dict["relation_labels"]  # (1, num_spans, nR)
        coref_labels = output_dict["coref_labels"]
        relation_scores = output_dict["relation_scores"]  # (1, num_spans_to_keep, max_antecedent)

        output_dict['loss'] = 0.0

        if relation_labels is not None:
            gold_labels = self._compute_antecedent_gold_labels(relation_labels, coref_labels)

            para_idx = output_dict['spans_para_idx'] #(1, Ns)
            para_pair = torch.abs(para_idx.unsqueeze(-1) - para_idx.unsqueeze(1))
            assert para_pair.shape == gold_labels.shape

            relation_scores = relation_scores.squeeze(-1)

            para_pair = (para_pair <= self._relation_coverage).long().float()

            sample_weight = ((3 * gold_labels) + 1)
            gold_labels_filtered = (gold_labels * para_pair)
            relation_scores_filtered = (relation_scores * para_pair)

            sample_weight_filtered = (sample_weight * para_pair)

            if (gold_labels > 1).byte().any() or (relation_scores > 1).byte().any():
                breakpoint()
            if (gold_labels < 0).byte().any() or (relation_scores < 0).byte().any():
                breakpoint()

            output_dict["loss"] = (
                F.binary_cross_entropy(relation_scores_filtered, gold_labels_filtered, reduction="none") * sample_weight_filtered
            ).sum() / output_dict['batch_size']

            output_dict["gold_relation_labels"] = gold_labels_filtered
            self._coref_scores(relation_scores_filtered, gold_labels_filtered)

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):
        new_output_dict = {"spans": [], "relation_scores": []}
        if "spans" in output_dict:
            new_output_dict["spans"] = output_dict["spans"].squeeze(0).detach().cpu().numpy()
            new_output_dict["relation_scores"] = (
                output_dict["relation_scores"].squeeze(0).squeeze(-1).detach().cpu().numpy()
            )
        if "gold_relation_labels" in output_dict:
            new_output_dict["relation_labels"] = output_dict["gold_relation_labels"].squeeze(0).detach().cpu().numpy()
        new_output_dict["metadata"] = output_dict["metadata"]
        return new_output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self._coref_scores.get_metric(reset)
        return {"rel_" + k: v for k, v in metrics.items()}

    def _compute_span_pair_embeddings(self, spans: torch.IntTensor, top_span_embeddings: torch.FloatTensor):
        """
        TODO(dwadden) document me and add comments.
        """
        # Shape: (batch_size, num_spans_to_keep, num_spans_to_keep, embedding_size)
        num_candidates = top_span_embeddings.size(1)

        embeddings_1_expanded = top_span_embeddings.unsqueeze(2)
        embeddings_1_tiled = embeddings_1_expanded.repeat(1, 1, num_candidates, 1)

        embeddings_2_expanded = top_span_embeddings.unsqueeze(1)
        embeddings_2_tiled = embeddings_2_expanded.repeat(1, num_candidates, 1, 1)

        similarity_embeddings = embeddings_1_expanded * embeddings_2_expanded

        pair_embeddings_list = [embeddings_1_tiled, embeddings_2_tiled, similarity_embeddings]
        pair_embeddings = torch.cat(pair_embeddings_list, dim=3)

        return pair_embeddings

    @staticmethod
    def _compute_antecedent_gold_labels(relation_labels: torch.IntTensor, coref_labels: torch.IntTensor):
        # Shape: (batch_size, num_spans_to_keep, max_antecedents)
        source_labels = relation_labels.unsqueeze(1)
        target_labels = relation_labels.unsqueeze(2)
        relation_indicator = (target_labels * source_labels).sum(-1).clamp(0, 1).float()

        source_labels = coref_labels.unsqueeze(1)
        target_labels = coref_labels.unsqueeze(2)
        coref_indicator = (target_labels * source_labels).sum(-1).clamp(0, 1).float()

        label = relation_indicator * (relation_indicator - coref_indicator)
        assert (label < 0).sum() == 0, breakpoint()

        return label

    @staticmethod
    def _flatten_spans(span_ix, spans_batched):
        """
        Spans are input with each minibatch as a sentence. For coref, it's easier to flatten them out
        and consider all sentences together as a document.
        """

        spans_flat = spans_batched.view(-1, 2)
        span_para_idx = torch.arange(0, spans_batched.shape[0]).unsqueeze(1).repeat(1, spans_batched.shape[1]).to(spans_flat.device)
        span_para_idx = span_para_idx.view(-1)[span_ix].unsqueeze(0)
        spans_flat = spans_flat[span_ix].unsqueeze(0)

        return spans_flat, span_para_idx

    @staticmethod
    def _flatten_span_info(span_info_batched, span_ix):
        feature_size = span_info_batched.size(-1)
        emb_flat = span_info_batched.view(-1, feature_size)
        span_info_flat = emb_flat[span_ix].unsqueeze(0)
        return span_info_flat

