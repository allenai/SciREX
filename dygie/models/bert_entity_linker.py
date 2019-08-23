from typing import Dict, Optional, Union, Any

import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import RegularizerApplicator
from allennlp.modules import FeedForward
from allennlp.nn.initializers import InitializerApplicator
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder

from dygie.training.thresholding_f1_metric import BinaryThresholdF1


@Model.register("bert_for_entity_linking")
class BertForClassification(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        bert_model: PretrainedBertEmbedder,
        aggregate_feedforward: FeedForward,
        dropout: float = 0.0,
        index: str = "bert",
        label_namespace: str = "labels",
        featured: bool = True,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super().__init__(vocab, regularizer)

        self.bert_model = bert_model.bert_model

        self._label_namespace = label_namespace

        self._dropout = torch.nn.Dropout(p=dropout)

        self._classification_layer = aggregate_feedforward
        self._loss = torch.nn.CrossEntropyLoss()
        self._index = index
        self._featured = featured

        self._f1 = BinaryThresholdF1()

        initializer(self._classification_layer)

    def forward(
        self,  # type: ignore
        tokens: Dict[str, torch.LongTensor],
        pair_features: torch.Tensor,
        label: torch.IntTensor = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        input_ids = tokens[self._index]
        token_type_ids = tokens[f"{self._index}-type-ids"]
        input_mask = (input_ids != 0).long()

        _, pooled = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=input_mask)

        pooled = self._dropout(pooled)

        # apply classification layer
        label_logits = self._classification_layer(torch.cat([pooled, pair_features], dim=-1) if self._featured else pooled)

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
