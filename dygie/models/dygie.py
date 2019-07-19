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
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util

# Import submodules.
from dygie.models.coref import CorefResolver
from dygie.models.ner import NERTagger
from dygie.models.relation import RelationExtractor

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("dygie")
class DyGIE(Model):
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
        If provided, will be used to calculate the regularization penalty during training.
    display_metrics: ``List[str]``. A list of the metrics that should be printed out during model
        training.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        context_layer: Seq2SeqEncoder,
        modules,  
        feature_size: int,
        max_span_width: int,
        loss_weights: Dict[str, int],
        lexical_dropout: float = 0.2,
        use_attentive_span_extractor: bool = True,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
        display_metrics: List[str] = None,
    ) -> None:
        super(DyGIE, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
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
        if use_attentive_span_extractor:
            self._attentive_span_extractor = SelfAttentiveSpanExtractor(input_dim=text_field_embedder.get_output_dim())
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
    def forward(self, text, spans, ner_labels, coref_labels, relation_labels, metadata):
        """
        TODO(dwadden) change this.
        """

        # In AllenNLP, AdjacencyFields are passed in as floats. This fixes it.
        relation_labels = relation_labels.long()

        # Shape: (batch_size, max_sentence_length, embedding_size)
        text_embeddings = self._lexical_dropout(self._text_field_embedder(text))

        # Shape: (batch_size, max_sentence_length)
        text_embeddings, text_mask, sentence_lengths = self.extract_sentence_from_context(metadata, text_embeddings)

        # Shape: (batch_size, max_sentence_length, encoding_dim)
        contextualized_embeddings = self._context_layer(text_embeddings, text_mask)
        assert spans.max() < contextualized_embeddings.shape[1]

        if self._attentive_span_extractor is not None:
            # Shape: (batch_size, num_spans, emebedding_size)
            attended_span_embeddings = self._attentive_span_extractor(text_embeddings, spans)

        # Shape: (batch_size, num_spans)
        span_mask = (spans[:, :, 0] >= 0).float()
        # SpanFields return -1 when they are used as padding. As we do
        # some comparisons based on span widths when we attend over the
        # span representations that we generate from these indices, we
        # need them to be <= 0. This is only relevant in edge cases where
        # the number of spans we consider after the pruning stage is >= the
        # total number of spans, because in this case, it is possible we might
        # consider a masked span.
        # Shape: (batch_size, num_spans, 2)
        spans = F.relu(spans.float()).long()

        # Shape: (batch_size, num_spans, 2 * encoding_dim + feature_size)
        endpoint_span_embeddings = self._endpoint_span_extractor(contextualized_embeddings, spans)

        if self._attentive_span_extractor is not None:
            # Shape: (batch_size, num_spans, emebedding_size + 2 * encoding_dim + feature_size)
            span_embeddings = torch.cat([endpoint_span_embeddings, attended_span_embeddings], -1)
        else:
            span_embeddings = endpoint_span_embeddings

        # Make calls out to the modules to get results.
        output_coref = {"loss": 0}
        output_ner = {"loss": 0}
        output_relation = {"loss": 0}

        # Prune and compute span representations for coreference module
        if self._loss_weights["coref"] > 0 or self._coref.coref_prop > 0:
            output_coref, coref_indices = self._coref.compute_representations(
                spans, span_mask, span_embeddings, sentence_lengths, coref_labels, metadata
            )

        # Prune and compute span representations for relation module
        if self._loss_weights["relation"] > 0 or self._relation.rel_prop > 0:
            output_relation = self._relation.compute_representations(
                spans, span_mask, span_embeddings, sentence_lengths, relation_labels, metadata
            )

        # Propagation of global information to enhance the span embeddings
        if self._coref.coref_prop > 0:
            # TODO(Ulme) Implement Coref Propagation
            output_coref = self._coref.coref_propagation(output_coref)
            span_embeddings = self._coref.update_spans(output_coref, span_embeddings, coref_indices)

        if self._relation.rel_prop > 0:
            output_relation = self._relation.relation_propagation(output_relation)
            span_embeddings = self.update_span_embeddings(
                span_embeddings,
                span_mask,
                output_relation["top_span_embeddings"],
                output_relation["top_span_mask"],
                output_relation["top_span_indices"],
            )

        # Make predictions and compute losses for each module
        if self._loss_weights["ner"] > 0:
            output_ner = self._ner(spans, span_mask, span_embeddings, sentence_lengths, ner_labels, metadata)

        if self._loss_weights["coref"] > 0:
            output_coref = self._coref.predict_labels(output_coref, metadata)

        if self._loss_weights["relation"] > 0:
            output_relation = self._relation.predict_labels(relation_labels, output_relation, metadata)

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

    def update_span_embeddings(self, span_embeddings, span_mask, top_span_embeddings, top_span_mask, top_span_indices):
        # TODO(Ulme) Speed this up by tensorizing

        new_span_embeddings = span_embeddings.clone()
        for sample_nr in range(len(top_span_mask)):
            for top_span_nr, span_nr in enumerate(top_span_indices[sample_nr]):
                if top_span_mask[sample_nr, top_span_nr] == 0 or span_mask[sample_nr, span_nr] == 0:
                    break
                new_span_embeddings[sample_nr, span_nr] = top_span_embeddings[sample_nr, top_span_nr]
        return new_span_embeddings

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
        # TODO(dwadden) which things are already decoded?
        res = {}
        if self._loss_weights["coref"] > 0:
            res["coref"] = self._coref.decode(output_dict["coref"])# TODO(dwadden) Add type.
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
