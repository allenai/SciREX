import logging
from itertools import product, combinations
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, TimeDistributed
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from overrides import overrides
from scripts.entity_utils import used_entities

from dygie.training.n_ary_relation_metrics import NAryRelationMetrics
from dygie.training.thresholding_f1_metric import BinaryThresholdF1

from collections import defaultdict

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class ClusterClassifier(Model):
    def __init__(
        self,
        vocab: Vocabulary = None,
        antecedent_feedforward: FeedForward = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super(ClusterClassifier, self).__init__(vocab, regularizer)

        self._antecedent_feedforward = TimeDistributed(antecedent_feedforward)
        self._antecedent_scorer = torch.nn.Linear(antecedent_feedforward.get_output_dim(), 2)

        self._global_scores = NAryRelationMetrics()

        initializer(self)

    def forward(self, **kwargs):
        raise NotImplementedError

    def compute_representations(
        self,  # type: ignore
        spans: torch.IntTensor,  # (1, Ns, 2)
        span_mask,
        span_embeddings,  # (1, Ns, E)
        coref_labels: torch.IntTensor,  # (1, Ns, C)
        cluster_labels: torch.IntTensor,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ

        cluster_span_embeddings = util.masked_mean(
            span_embeddings.unsqueeze(2), coref_labels.unsqueeze(-1), dim=1
        )  # (P, C, E)

        paragraph_cluster_mask = (coref_labels.sum(1) > 0).float().unsqueeze(-1)  # (P, C, 1)

        paragraph_cluster_embeddings = cluster_span_embeddings * paragraph_cluster_mask

        assert (
            paragraph_cluster_embeddings.shape[1] == coref_labels.shape[2]
            and paragraph_cluster_embeddings.shape[2] == span_embeddings.shape[-1]
        )

        relation_scores, relation_logits = self.get_relation_scores(paragraph_cluster_embeddings)  # (1, R')

        output_dict = {}
        output_dict["doc_id"] = metadata[0]["doc_key"]
        output_dict["metadata"] = metadata
        output_dict["relation_scores"] = relation_scores
        output_dict["relation_logits"] = relation_logits
        output_dict["cluster_labels"] = cluster_labels

        if cluster_labels is not None:
            output_dict = self.predict_labels(
                relation_scores, relation_logits, cluster_labels, output_dict
            )

        return output_dict

    def get_relation_scores(self, paragraph_cluster_embeddings):
        # (B, NS, NS, E)
        relation_embeddings = self._antecedent_feedforward(paragraph_cluster_embeddings) #(P, R, e)
        relation_embeddings = relation_embeddings.mean(0)
        relation_logits = self._antecedent_scorer(relation_embeddings).squeeze(-1)
        relation_scores = torch.nn.Softmax(dim=-1)(relation_logits)[:, 1]
        return relation_scores, relation_logits

    def predict_labels(self, relation_scores, relation_logits, relation_labels, output_dict):
        output_dict["loss"] = 0.0

        if relation_labels is not None:
            assert (relation_scores <= 1.0).all() & (relation_scores >= 0.0).all()
            assert (relation_labels <= 1.0).all() & (relation_labels >= 0.0).all()

            output_dict["loss"] = F.cross_entropy(
                relation_logits,
                relation_labels.long(),
                reduction="mean",
                weight=torch.Tensor([1.0, 10.0]).to(relation_logits.device),
            )

            self._global_scores(
                [[x] for x in list(range(len(relation_scores)))],
                list(output_dict["cluster_labels"].cpu().data.numpy()),
                relation_scores,
                output_dict["doc_id"],
            )

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):
        new_output_dict = {
            "scores": output_dict.get("relation_scores", np.array([])),
            "metadata" : output_dict['metadata']
        }

        if len(new_output_dict["scores"]) > 0:
            new_output_dict["scores"] = new_output_dict["scores"].cpu().data.numpy()

        return new_output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        global_metrics = self._global_scores.get_metric(reset)
        metrics.update(global_metrics)
        return {"cluster_" + k: v for k, v in metrics.items()}
