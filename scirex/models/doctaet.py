from typing import Dict, Optional, Union, Any

import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import RegularizerApplicator
from allennlp.modules import FeedForward
from allennlp.nn.initializers import InitializerApplicator
from scirex.models.bert_token_embedder_modified import PretrainedBertEmbedder

from scirex.metrics.thresholding_f1_metric import BinaryThresholdF1


@Model.register("doctaet")
class DoctaetModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        bert_model: PretrainedBertEmbedder,
        aggregate_feedforward: FeedForward,
        dropout: float = 0.0,
        index: str = "bert",
        label_namespace: str = "labels",
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super().__init__(vocab, regularizer)

        self.bert_model = bert_model.bert_model

        self._label_namespace = label_namespace
        print(list(self.vocab._retained_counter["labels"].items()))

        label_vocab = self.vocab.get_index_to_token_vocabulary('labels')

        total_size = sum(self.vocab._retained_counter['labels'].values())
        self._class_weight = [0] * len(label_vocab)
        for i, t in label_vocab.items() :
            self._class_weight[i] = total_size / self.vocab._retained_counter['labels'][t]
        
        self._dropout = torch.nn.Dropout(p=dropout)
        self._pos_index = self.vocab.get_token_to_index_vocabulary(label_namespace)['True']

        self._classification_layer = aggregate_feedforward
        self._loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(self._class_weight))
        self._index = index

        self._f1 = BinaryThresholdF1()

        initializer(self._classification_layer)

    def forward(
        self,  # type: ignore
        tokens: Dict[str, torch.LongTensor],
        label: torch.IntTensor = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        input_ids = tokens[self._index]
        token_type_ids = tokens[f"{self._index}-type-ids"]
        input_mask = (input_ids != 0).long()

        _, pooled = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=input_mask)

        label_logits = self._classification_layer(self._dropout(pooled))
        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

        output_dict = {"label_probs": label_probs[..., self._pos_index]}

        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            self._f1(label_probs[..., self._pos_index], label.long().view(-1))
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
