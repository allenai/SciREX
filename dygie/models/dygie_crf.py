import copy
import logging
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.span_extractors import EndpointSpanExtractor, SelfAttentiveSpanExtractor
from dygie.models.span_extractor import MaxPoolSpanExtractor
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util

# Import submodules.
from dygie.models.coref import CorefResolver
from dygie.models.ner_crf_tagger import NERTagger
from dygie.models.relation_pwc import RelationExtractor

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("dygie_crf")
class DyGIECRF(Model):
    """
    TODO(dwadden) document me.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``text`` ``TextField`` we get as input to the model.
    context_layer : ``Seq2SeqEncoder``
        This layer incorporates contextual information for each word in the document.
    feature_size: ``int``
        The embedding size for all the embedded features, such as distances or span widths.
    submodule_params: ``TODO(dwadden)``
        A nested dictionary specifying parameters to be passed on to initialize submodules.
    max_span_width: ``int``
        The maximum width of candidate spans.
    lexical_dropout: ``int``
        The probability of dropping out dimensions of the embedded text.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the reglarization penalty during training.
    display_metrics: ``List[str]``. A list of the metrcs that should be printed out during model
        training.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        residual_text_field_embedder: TextFieldEmbedder,
        context_layer: Seq2SeqEncoder,
        modules,
        feature_size: int,
        loss_weights: Dict[str, int],
        max_span_width: int = 20,
        lexical_dropout: float = 0.2,
        use_attentive_span_extractor: bool = True,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
        display_metrics: List[str] = None,
    ) -> None:
        super(DyGIECRF, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._residual_text_field_embedder = residual_text_field_embedder
        self._context_layer = context_layer

        self._loss_weights = loss_weights.as_dict()
        self._permanent_loss_weights = copy.deepcopy(self._loss_weights)

        self._coref = CorefResolver.from_params(vocab=vocab, feature_size=feature_size, params=modules.pop("coref"))
        self._ner = NERTagger.from_params(vocab=vocab, params=modules.pop("ner"))
        self._relation = RelationExtractor.from_params(
            vocab=vocab, feature_size=feature_size, params=modules.pop("relation")
        )

        self._endpoint_span_extractor = EndpointSpanExtractor(
            context_layer.get_output_dim(),
            combination="x,y",
            num_width_embeddings=max_span_width,
            span_width_embedding_dim=feature_size,
            bucket_widths=False,
        )

        self._residual_span_extractor = MaxPoolSpanExtractor(residual_text_field_embedder.get_output_dim())

        if use_attentive_span_extractor:
            self._attentive_span_extractor = SelfAttentiveSpanExtractor(
                input_dim=text_field_embedder.get_output_dim() + residual_text_field_embedder.get_output_dim()
            )
        else:
            self._attentive_span_extractor = None

        self._max_span_width = max_span_width

        self._display_metrics = display_metrics

        if lexical_dropout > 0:
            self._lexical_dropout = torch.nn.Dropout(p=lexical_dropout)
        else:
            self._lexical_dropout = lambda x: x

        initializer(self)

    @overrides
    def forward(self, text, ner_labels, ner_entity_labels, ner_link_labels, ner_is_entity_labels, metadata):

        # Shape: (batch_size, max_sentence_length, embedding_size)
        text_embeddings = self._lexical_dropout(self._text_field_embedder(text))
        residual_text_embeddings = self._lexical_dropout(self._residual_text_field_embedder(text))

        text_embeddings = torch.cat([text_embeddings, residual_text_embeddings], dim=-1)
        text_embeddings, text_mask, sentence_lengths = self.extract_sentence_from_context(metadata, text_embeddings)

        # Shape: (batch_size, max_sentence_length, encoding_dim)
        contextualized_embeddings = self._context_layer(text_embeddings, text_mask)

        # Make calls out to the modules to get results.
        output_coref = {"loss": 0}
        output_ner = {"loss": 0}
        output_relation = {"loss": 0}

        ner_labels_dispatcher = {
            "ner_labels": ner_labels,
            "ner_entity_labels": ner_entity_labels,
            "ner_link_labels": ner_link_labels,
            "ner_is_entity_labels" : ner_is_entity_labels
        }

        # Make predictions and compute losses for each module
        output_ner = self._ner(
            contextualized_embeddings, text_mask, ner_labels_dispatcher[self._ner.label_namespace], metadata
        )

        if "loss" not in output_coref:
            output_coref["loss"] = 0
        if "loss" not in output_relation:
            output_relation["loss"] = 0

        loss = (
            self._loss_weights["coref"] * output_coref["loss"]
            + self._loss_weights["ner"] * output_ner["loss"]
            + self._loss_weights["relation"] * output_relation["loss"]
        )

        output_dict = dict(coref=output_coref, relation=output_relation, ner=output_ner)
        output_dict["loss"] = loss

        return output_dict

    def extract_sentence_from_context(self, metadata, text_embeddings):
        sentence_spans = torch.LongTensor([[x["start_ix"], x["end_ix"]] for x in metadata]).to(text_embeddings.device)
        sentence_lengths = sentence_spans[:, 1] - sentence_spans[:, 0]
        max_sentence_length = sentence_lengths.max()

        range_vector = util.get_range_vector(max_sentence_length, util.get_device_of(text_embeddings)).view(1, -1)
        span_indices = torch.clamp_max(sentence_spans[:, 0:1] + range_vector, text_embeddings.shape[1] - 1)
        text_mask = util.get_mask_from_sequence_lengths(sentence_lengths, max_length=max_sentence_length)
        text_embeddings = util.batched_index_select(text_embeddings, span_indices) * text_mask.unsqueeze(-1).float()
        return text_embeddings, text_mask, sentence_lengths

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
        res = {}
        if self._loss_weights["coref"] > 0:
            res["coref"] = self._coref.decode(output_dict["coref"])
        if self._loss_weights["ner"] > 0:
            res["ner"] = self._ner.decode(output_dict["ner"])
        if self._loss_weights["relation"] > 0:
            res["relation"] = self._relation.decode(output_dict["relation"])

        return res

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        Get all metrics from all modules. For the ones that shouldn't be displayed, prefix their
        keys with an underscore.
        """
        metrics_coref = self._coref.get_metrics(reset=reset)
        metrics_ner = self._ner.get_metrics(reset=reset)
        metrics_relation = self._relation.get_metrics(reset=reset)

        # Make sure that there aren't any conflicting names.
        metric_names = list(metrics_coref.keys()) + list(metrics_ner.keys()) + list(metrics_relation.keys())
        assert len(set(metric_names)) == len(metric_names)
        all_metrics = dict(list(metrics_coref.items()) + list(metrics_ner.items()) + list(metrics_relation.items()))

        # If no list of desired metrics given, display them all.
        if self._display_metrics is None:
            return all_metrics
        # Otherwise only display the selected ones.
        res = {}
        for k, v in all_metrics.items():
            if k in self._display_metrics:
                res[k] = v
            else:
                new_k = "_" + k
                res[new_k] = v
        return res
