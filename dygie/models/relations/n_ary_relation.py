import logging
from itertools import product
from typing import Any, Dict, List, Optional, Tuple
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

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class RelationExtractor(Model):
    def __init__(
        self,
        vocab: Vocabulary = None,
        antecedent_feedforward: FeedForward = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super(RelationExtractor, self).__init__(vocab, regularizer)

        self._antecedent_feedforward = TimeDistributed(antecedent_feedforward)
        self._antecedent_scorer = TimeDistributed(torch.nn.Linear(antecedent_feedforward.get_output_dim(), 1))

        self._coref_scores = BinaryThresholdF1()
        self._global_scores = NAryRelationMetrics()

        initializer(self)

    def forward(self, **kwargs) :
        raise NotImplementedError

    @staticmethod
    def generate_product(
        type_map: Dict[str, List[int]],
        relation_index: Dict[int, List[int]],
        coref_labels: torch.LongTensor,
        device: int,
    ):
        n_rel = []
        n_rel_label = []

        coref_labels_nnz = set(list(coref_labels.sum(1).squeeze(0).nonzero().squeeze(1).cpu().numpy()))
        type_lists = [type_map[x] for x in used_entities]
        for clist in product(*type_lists):
            if len(set(clist) - coref_labels_nnz) == 0:
                n_rel.append(clist)
                n_rel_label.append(1 if relation_index is not None and clist in relation_index.values() else 0)

        n_rel_tensor = torch.LongTensor(n_rel)
        n_rel_label = torch.LongTensor(n_rel_label)
        return n_rel, n_rel_tensor.to(device), n_rel_label.to(device)

    def compute_representations(
        self,  # type: ignore
        spans: torch.IntTensor,
        span_embeddings,
        coref_labels: torch.IntTensor,
        cluster_type_dict: Dict[str, List[int]],
        relation_index_dict: Dict[int, List[int]] = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        
        if span_embeddings.nelement() == 0:
            breakpoint()

        cluster_embeddings = util.masked_max(span_embeddings.unsqueeze(2), coref_labels.unsqueeze(-1), dim=1)
        assert (
            cluster_embeddings.shape[1] == coref_labels.shape[2]
            and cluster_embeddings.shape[2] == span_embeddings.shape[-1]
        )


        n_rel_list, n_rel, relation_labels = self.generate_product(
            cluster_type_dict, relation_index_dict, coref_labels, device=cluster_embeddings.device
        )

        if len(n_rel_list) == 0:
            return {"loss": 0.0}

        n_rel = n_rel.unsqueeze(0)  # (1, R', n)
        relation_labels = relation_labels.unsqueeze(0)
        all_relation_embeddings = util.batched_index_select(cluster_embeddings, n_rel)  # (1, R', n, E)

        relation_scores = self.get_relation_scores(all_relation_embeddings)  # (1, R')
        output_dict = {}
        output_dict["relations_candidates_list"] = n_rel_list
        output_dict["relations_true_list"] = list(relation_index_dict.values() if relation_index_dict is not None else [])
        output_dict["relations"] = n_rel
        output_dict["relation_labels"] = relation_labels
        output_dict["doc_id"] = metadata[0]["doc_key"]
        output_dict["metadata"] = metadata
        output_dict["relation_scores"] = relation_scores

        if relation_index_dict is not None :
            output_dict = self.predict_labels(relation_scores, relation_labels, output_dict)

        return output_dict

    def get_relation_scores(self, relation_embeddings):
        # (B, NS, NS, E)
        relation_embeddings = relation_embeddings.view(relation_embeddings.shape[0], relation_embeddings.shape[1], -1)
        relation_scores = torch.sigmoid(
            self._antecedent_scorer(self._antecedent_feedforward(relation_embeddings))
        ).squeeze(-1)
        return relation_scores

    def predict_labels(self, relation_scores, relation_labels, output_dict):
        output_dict["loss"] = 0.0

        if relation_labels is not None:
            assert (relation_scores <= 1.0).all() & (relation_scores >= 0.0).all()
            assert (relation_labels <= 1.0).all() & (relation_labels >= 0.0).all()
            output_dict["loss"] = F.binary_cross_entropy(relation_scores, relation_labels.float(), reduction="sum")
            self._coref_scores(relation_scores, relation_labels)
            self._global_scores(
                output_dict["relations_candidates_list"],
                output_dict["relations_true_list"],
                relation_scores,
                output_dict["doc_id"],
            )

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):
        new_output_dict = {
            "candidates" : output_dict.get('relations_candidates_list', []),
            "gold" : output_dict.get('relations_true_list', []),
            "scores" : output_dict.get('relation_scores', np.array([]))
        }

        if len(new_output_dict['scores']) > 0 :
            new_output_dict['scores'] = new_output_dict['scores'].cpu().data.numpy()

        return new_output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self._coref_scores.get_metric(reset)
        global_metrics = self._global_scores.get_metric(metrics["threshold"], reset)
        metrics.update({"global_" + k: v for k, v in global_metrics.items()})
        return {"n_ary_rel_" + k: v for k, v in metrics.items()}
