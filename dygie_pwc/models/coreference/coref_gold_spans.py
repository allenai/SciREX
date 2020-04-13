import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, TimeDistributed
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from overrides import overrides

from dygie_pwc.models import shared
from dygie_pwc.training.thresholding_f1_metric import BinaryThresholdF1

from dygie_pwc.data.dataset_readers.span_utils import span_feature_size

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class CorefResolverCRF(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        antecedent_feedforward: FeedForward,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super(CorefResolverCRF, self).__init__(vocab, regularizer)

        self._antecedent_feedforward = TimeDistributed(antecedent_feedforward)
        self._basic_normalizer = torch.nn.Linear(
            antecedent_feedforward.get_output_dim() * 2, antecedent_feedforward.get_output_dim(), bias=True
        )

        self._bias = torch.nn.Linear(antecedent_feedforward.get_output_dim() + span_feature_size, 1)

        self._coref_scores = BinaryThresholdF1()

        initializer(self)

    def forward(self, **kwargs):
        raise NotImplementedError

    def compute_representations(
        self,  # type: ignore
        spans_batched,
        span_mask_batched,
        span_embeddings_batched,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ

        span_ix = span_mask_batched.view(-1).nonzero().squeeze()  # Indices of the spans to keep.
        spans = self._flatten_span_info(spans_batched, span_ix)
        span_embeddings = self._flatten_span_info(span_embeddings_batched, span_ix)  # (1, Ns, E)

        span_embeddings = self._antecedent_feedforward(span_embeddings).squeeze(0)
        Ns = span_embeddings.shape[0]
        span_embeddings_concatenated = torch.cat(
            [span_embeddings.unsqueeze(0).expand(Ns, -1, -1), span_embeddings.unsqueeze(1).expand(-1, Ns, -1)], dim=-1
        )
        span_embeddings_concatenated = self._basic_normalizer(span_embeddings_concatenated).squeeze(0)

        # coreference_scores = torch.bmm(span_embeddings, span_embeddings.transpose(1, 2)).squeeze(0)

        a, b = metadata[0]["span_idx"]
        coreference_labels = torch.Tensor(metadata[0]["document_metadata"]["coref_labels"][a:b, a:b]).to(
            span_embeddings.device
        )
        coreference_mask = torch.Tensor(metadata[0]["document_metadata"]["coref_mask"][a:b, a:b]).to(
            span_embeddings.device
        )

        coreference_features = torch.Tensor(metadata[0]["document_metadata"]["coref_features"][a:b, a:b]).to(
            span_embeddings.device
        )

        # breakpoint()
        coreference_scores = torch.cat([span_embeddings_concatenated, coreference_features], dim=-1)
        coreference_scores = torch.sigmoid(self._bias(coreference_scores).squeeze(-1))

        assert (coreference_scores > 1.0001).sum() == 0, breakpoint()
        assert (coreference_scores < 0.0).sum() == 0, breakpoint()

        assert coreference_scores.shape == coreference_labels.shape, breakpoint()
        assert coreference_scores.shape == coreference_mask.shape, breakpoint()

        output_dict = {
            "coreference_pair_scores": coreference_scores,
            "coreference_pair_labels": coreference_labels,
            "coreference_pair_mask": coreference_mask,
            "spans_batched": spans_batched,
            "spans_ix": span_ix,
            "span_mask_batched": span_mask_batched,
            "metadata": metadata,
        }

        return self.predict_labels(output_dict)

    def predict_labels(self, output_dict):
        coreference_pair_scores = output_dict["coreference_pair_scores"].view(-1)  # (Ns, Ns)

        if "coreference_pair_labels" in output_dict:
            coreference_pair_labels = output_dict["coreference_pair_labels"].view(-1)
            coreference_pair_mask = output_dict["coreference_pair_mask"].view(-1).bool()

            valid_scores = coreference_pair_scores[coreference_pair_mask]
            valid_labels = coreference_pair_labels[coreference_pair_mask]

            assert len(valid_scores) == coreference_pair_mask.sum(), breakpoint()

            if valid_labels.nelement() == 0:
                output_dict["loss"] = 0.0
            else:
                output_dict["loss"] = F.binary_cross_entropy(valid_scores, valid_labels)

                self._coref_scores(valid_scores, valid_labels.long())

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self._coref_scores.get_metric(reset)
        return {"coref_" + k: v for k, v in metrics.items()}

    @staticmethod
    def _flatten_spans(span_ix, spans_batched, sentence_lengths):
        sentence_offset = shared.cumsum_shifted(sentence_lengths).unsqueeze(1).unsqueeze(2)
        assert len(sentence_offset.shape) == 3 and sentence_offset.size(1) == 1 and sentence_offset.size(2) == 1
        spans_offset = spans_batched + sentence_offset
        spans_flat = spans_offset.view(-1, 2)
        spans_flat = spans_flat[span_ix].unsqueeze(0)

        return spans_flat

    @staticmethod
    def _flatten_span_info(span_info_batched, span_ix):
        feature_size = span_info_batched.size(-1)
        emb_flat = span_info_batched.view(-1, feature_size)
        span_info_flat = emb_flat[span_ix].unsqueeze(0)
        return span_info_flat
