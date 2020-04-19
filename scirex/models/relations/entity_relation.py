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
from scirex_utilities.entity_utils import used_entities

from scirex.metrics.n_ary_relation_metrics import NAryRelationMetrics
from scirex.metrics.thresholding_f1_metric import BinaryThresholdF1

from collections import defaultdict

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class RelationExtractor(Model):
    def __init__(
        self,
        vocab: Vocabulary = None,
        antecedent_feedforward: FeedForward = None,
        relation_cardinality: int = 2,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super(RelationExtractor, self).__init__(vocab, regularizer)

        self._antecedent_feedforward = TimeDistributed(antecedent_feedforward)
        self._antecedent_scorer = TimeDistributed(torch.nn.Linear(antecedent_feedforward.get_output_dim(), 1))
        self._span_embedding_size = antecedent_feedforward.get_input_dim() // 4
        self._bias_vectors = torch.nn.Parameter(torch.zeros((1, 4, self._span_embedding_size)))

        self._relation_cardinality = relation_cardinality

        self._pos_weight_dict = {
            2: 1.0, 3: 1.0, 4: 3.3
        }

        self._pos_weight = self._pos_weight_dict[relation_cardinality]

        self._relation_type_map = {
            tuple(e): i for i, e in enumerate(combinations(used_entities, self._relation_cardinality))
        }

        self._binary_scores = BinaryThresholdF1()
        self._global_scores = NAryRelationMetrics()

        self.ignore_empty_clusters = False

        initializer(self)

    def map_cluster_to_type_embeddings(self, type_to_cluster_map: Dict[str, List[int]]):
        cluster_to_type_map = {v: used_entities.index(k) for k, vs in type_to_cluster_map.items() for v in vs}

        n_cluster = max(cluster_to_type_map.keys()) + 1
        cluster_to_type_indices = [0] * n_cluster
        for v, t in cluster_to_type_map.items():
            cluster_to_type_indices[v] = t

        return self._bias_vectors[:, cluster_to_type_indices]

    def forward(self, **kwargs):
        raise NotImplementedError

    def generate_product(
        self,
        type_to_clusters_map: Dict[str, List[int]],
        n_true_clusters: int,
        relation_to_clusters_map: Dict[int, List[int]] = None,
        cluster_to_size_map: Dict[str, int] = None
    ):
        bias_vectors_clusters = {x: i + n_true_clusters for i, x in enumerate(used_entities)}
        candidate_relations = []
        candidate_relations_labels = []
        candidate_relations_types = []

        if relation_to_clusters_map is None:
            relation_to_clusters_map = {}

        cluster_to_relations_map = defaultdict(set)
        for r, clist in relation_to_clusters_map.items():
            for t in bias_vectors_clusters.values():
                cluster_to_relations_map[t].add(r)
            for c in clist:
                cluster_to_relations_map[c].add(r)

        for e in combinations(used_entities, self._relation_cardinality):
            type_lists = [type_to_clusters_map[x] if x in e else [bias_vectors_clusters[x]] for x in used_entities]
            for clist in product(*type_lists):
                common_relations = set.intersection(*[cluster_to_relations_map[c] for c in clist])
                if self.ignore_empty_clusters and cluster_to_relations_map is not None :
                    if any(cluster_to_size_map[c] == 0 for c in clist if c < len(cluster_to_size_map)):
                        continue

                candidate_relations.append(clist)
                candidate_relations_labels.append(1 if len(common_relations) > 0 else 0)
                candidate_relations_types.append(self._relation_type_map[tuple(e)])

        return candidate_relations, candidate_relations_labels, candidate_relations_types

    def compute_representations(
        self,  # type: ignore
        spans: torch.IntTensor,  # (1, Ns, 2)
        span_mask,
        span_embeddings,  # (1, Ns, E)
        coref_labels: torch.IntTensor,  # (1, Ns, C)
        type_to_cluster_ids: Dict[str, List[int]],
        relation_to_cluster_ids: Dict[int, List[int]] = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ

        if coref_labels.sum() == 0:
            return {"loss": 0.0, "metadata" : metadata}

        cluster_to_size_map = coref_labels.sum(0).sum(0).cpu().data.numpy()
        try :
            cluster_type_embeddings = self.map_cluster_to_type_embeddings(type_to_cluster_ids)  # (1, C, E)
        except :
            breakpoint()

        sum_embeddings = (span_embeddings.unsqueeze(2) * coref_labels.float().unsqueeze(-1)).sum(1)
        length_embeddings =  (coref_labels.unsqueeze(-1).sum(1) + 1e-5)

        cluster_span_embeddings = sum_embeddings / length_embeddings

        paragraph_cluster_mask = (coref_labels.sum(1) > 0).float().unsqueeze(-1)  # (P, C, 1)

        try :
            paragraph_cluster_embeddings = cluster_span_embeddings * paragraph_cluster_mask + cluster_type_embeddings * (
                1 - paragraph_cluster_mask
            ) # (P, C, E)
        except :
            breakpoint()

        assert (
            paragraph_cluster_embeddings.shape[1] == coref_labels.shape[2]
            and paragraph_cluster_embeddings.shape[2] == span_embeddings.shape[-1]
        )

        paragraph_cluster_embeddings = torch.cat(
            [paragraph_cluster_embeddings, self._bias_vectors.expand(paragraph_cluster_embeddings.shape[0], -1, -1)],
            dim=1,
        )  # (P, C+4, E)
        n_true_clusters = coref_labels.shape[-1]

        candidate_relations, candidate_relations_labels, candidate_relations_types = self.generate_product(
            type_to_clusters_map=type_to_cluster_ids,
            relation_to_clusters_map=relation_to_cluster_ids,
            n_true_clusters=n_true_clusters,
            cluster_to_size_map=cluster_to_size_map
        )

        candidate_relations_tensor = torch.LongTensor(candidate_relations).to(span_embeddings.device)  # (R, 4)
        candidate_relations_labels_tensor = torch.LongTensor(candidate_relations_labels).to(
            span_embeddings.device
        )  # (R, )

        if len(candidate_relations) == 0:
            return {"loss": 0.0, "metadata" : metadata}

        all_relation_embeddings = util.batched_index_select(
            paragraph_cluster_embeddings,
            candidate_relations_tensor.unsqueeze(0).expand(paragraph_cluster_embeddings.shape[0], -1, -1),
        )  # (P, R', n, E)

        relation_scores, relation_logits = self.get_relation_scores(all_relation_embeddings)  # (1, R')
        output_dict = {}
        output_dict["relations_candidates_list"] = candidate_relations
        output_dict["cluster_to_size_map"] = cluster_to_size_map
        output_dict["relation_labels"] = candidate_relations_labels
        output_dict["relation_types"] = candidate_relations_types
        output_dict["doc_id"] = metadata[0]["doc_id"]
        output_dict["metadata"] = metadata
        output_dict["relation_scores"] = relation_scores
        output_dict["relation_logits"] = relation_logits

        if relation_to_cluster_ids is not None:
            output_dict = self.predict_labels(
                relation_scores, relation_logits, candidate_relations_labels_tensor, output_dict
            )

        return output_dict

    def get_relation_scores(self, relation_embeddings):
        # (B, NS, NS, E)
        relation_embeddings = relation_embeddings.view(relation_embeddings.shape[0], relation_embeddings.shape[1], -1) #(P, R, E*4)
        relation_embeddings = self._antecedent_feedforward(relation_embeddings) #(P, R, e)
        relation_embeddings = relation_embeddings.max(0, keepdim=True)[0]
        relation_logits = self._antecedent_scorer(relation_embeddings).squeeze(-1).squeeze(0)
        relation_scores = torch.sigmoid(relation_logits)
        return relation_scores, relation_logits

    def predict_labels(self, relation_scores, relation_logits, relation_labels, output_dict):
        output_dict["loss"] = 0.0

        if relation_labels is not None:
            assert (relation_scores <= 1.0).all() & (relation_scores >= 0.0).all(), breakpoint()
            assert (relation_labels <= 1.0).all() & (relation_labels >= 0.0).all(), breakpoint()

            output_dict["loss"] = F.binary_cross_entropy_with_logits(
                relation_logits,
                relation_labels.float(),
                reduction="mean",
                pos_weight=torch.Tensor([self._pos_weight]).to(relation_logits.device)
            )

            self._global_scores(
                output_dict["relations_candidates_list"],
                output_dict["relation_labels"],
                relation_scores,
                output_dict["doc_id"],
            )

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):
        new_output_dict = {
            "linked_clusters" : [i for i, x in enumerate(output_dict.get("cluster_to_size_map", [])) if x > 0],
            "candidates": output_dict.get("relations_candidates_list", []),
            "gold": output_dict.get("relations_true_list", []),
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
        metrics.update({"global_" + k: v for k, v in global_metrics.items()})
        return {"n_ary_rel_" + k: v for k, v in metrics.items()}
