import copy
import logging
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from allennlp.common.params import Params
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.span_extractors import EndpointSpanExtractor, SelfAttentiveSpanExtractor
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from allennlp.training.metrics import Average
from overrides import overrides

# Import submodules.
from dygie.models.coreference.coref_gold_spans import CorefResolverCRF
from dygie.models.relations.n_ary_relation import RelationExtractor as NAryRelationExtractor
from dygie.models.ner.ner_crf_tagger import NERTagger
from dygie.models.relations.relation_pwc_crf import RelationExtractor
from dygie.models.span_classifiers.span_classifier_binary import SpanClassifier

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("dygie_crf_n_ary")
class DyGIECRF(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        context_layer: Seq2SeqEncoder,
        modules: Params,
        loss_weights: Dict[str, int],
        lexical_dropout: float = 0.2,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
        display_metrics: List[str] = None,
    ) -> None:
        super(DyGIECRF, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._context_layer = context_layer
        self._lexical_dropout = torch.nn.Dropout(p=lexical_dropout)

        modules = Params(modules)

        self._ner = NERTagger.from_params(vocab=vocab, params=modules.pop("ner"))
        self._coref = CorefResolverCRF.from_params(vocab=vocab, params=modules.pop("coref"))
        self._link_classifier = SpanClassifier.from_params(vocab=vocab, params=modules.pop("link_classifier"))
        self._span_binary_relation = RelationExtractor.from_params(vocab=vocab, params=modules.pop("relation"))
        self._cluster_n_ary_relation = NAryRelationExtractor.from_params(
            vocab=vocab, params=modules.pop("n_ary_relation")
        )

        self._endpoint_span_extractor = EndpointSpanExtractor(context_layer.get_output_dim(), combination="x,y")
        self._attentive_span_extractor = SelfAttentiveSpanExtractor(input_dim=context_layer.get_output_dim())

        self._loss_weights = loss_weights
        self._permanent_loss_weights = copy.deepcopy(self._loss_weights)

        self._display_metrics = display_metrics
        self._multi_task_loss_metrics = {k: Average() for k in ["ner", "relation", "linked", "n_ary_relation", "coref"]}

        self.training_mode = True
        self.prediction_mode = False

        initializer(self)

    @overrides
    def forward(
        self,
        text,
        ner_is_entity_labels,
        ner_entity_labels,
        spans=None,
        span_coref_labels=None,
        span_link_labels=None,
        span_entity_labels=None,
        span_features=None,
        relation_index=None,
        metadata=None,
    ):

        output_embedding = self.embedding_forward(text)
        output_ner = self.ner_forward(output_embedding, ner_entity_labels, metadata)
        output_span_embeddings = self.span_embeddings_forward(
            output_embedding, spans, span_entity_labels, span_features, metadata
        )

        output_coref = self.coref_forward(output_span_embeddings, metadata)

        output_relation, output_linker, output_n_ary_relation = self.link_and_relation_forward(
            output_span_embeddings, metadata, span_link_labels, relation_index, span_coref_labels
        )

        loss = (
            self._loss_weights['ner'] * output_ner["loss"]
            + self._loss_weights['relation'] * output_relation["loss"]
            + self._loss_weights['linked'] * output_linker["loss"]
            + self._loss_weights['n_ary_relation'] * output_n_ary_relation["loss"]
            + self._loss_weights['coref'] * output_coref["loss"]
        )

        output_dict = dict(
            relation=output_relation,
            ner=output_ner,
            linked=output_linker,
            n_ary_relation=output_n_ary_relation,
            coref=output_coref,
        )

        output_dict["loss"] = loss
        for k in self._multi_task_loss_metrics:
            l = output_dict[k]["loss"]
            self._multi_task_loss_metrics[k](l)

        return output_dict

    def embedding_forward(self, text):
        # Shape: (batch_size, max_sentence_length, embedding_size)
        text_embeddings = self._lexical_dropout(self._text_field_embedder(text))
        text_mask = util.get_text_field_mask(text)
        sentence_lengths = text_mask.sum(-1)

        # # Shape: (total_sentence_length, encoding_dim)
        # flat_text_embeddings = text_embeddings.view(-1, text_embeddings.size(-1))
        # flat_text_mask = text_mask.view(-1).byte()

        # filtered_text_embeddings = flat_text_embeddings[flat_text_mask.bool()]
        # filtered_contextualized_embeddings = self._context_layer(
        #     filtered_text_embeddings.unsqueeze(0),
        #     torch.ones((1, filtered_text_embeddings.size(0)), device=filtered_text_embeddings.device).byte(),
        # ).squeeze(0)

        # flat_contextualized_embeddings = torch.zeros(
        #     (flat_text_embeddings.size(0), filtered_contextualized_embeddings.size(1)),
        #     device=filtered_text_embeddings.device,
        # )
        # flat_contextualized_embeddings.masked_scatter_(
        #     flat_text_mask.unsqueeze(-1).bool(), filtered_contextualized_embeddings
        # )

        # # Shape: (batch_size, max_sentence_length, embedding_size)
        # contextualized_embeddings = flat_contextualized_embeddings.reshape(
        #     (text_embeddings.size(0), text_embeddings.size(1), flat_contextualized_embeddings.size(-1))
        # )

        contextualized_embeddings = self._context_layer(text_embeddings, text_mask)

        output_embedding = {
            "contextualised": contextualized_embeddings,
            "text": text_embeddings,
            "mask": text_mask,
            "lengths": sentence_lengths,
        }
        return output_embedding

    def span_embeddings_forward(self, output_embedding, spans, span_entity_labels, span_features, metadata):
        output_span_embeddings = {"valid": False}

        if spans.nelement() != 0:
            span_mask, spans, span_embeddings = self.extract_span_embeddings(output_embedding["contextualised"], spans)

            if span_mask.sum() != 0:
                span_offset = self.offset_span_by_para_start(metadata, spans, span_mask)
                span_position = self.get_span_position(metadata, span_offset)
                span_entity_labels_one_hot = self.get_span_one_hot_labels(
                    "span_entity_labels", span_entity_labels, spans
                )

                span_features = torch.cat([span_position, span_entity_labels_one_hot, span_features.float()], dim=-1)
                featured_span_embeddings = torch.cat([span_embeddings, span_features], dim=-1)
                span_ix = span_mask.view(-1).nonzero().squeeze(1).long()

                output_span_embeddings = {
                    "span_mask": span_mask,
                    "span_ix": span_ix,
                    "spans": self._flatten_span_info(span_offset, span_ix),
                    "span_embeddings": self._flatten_span_info(span_embeddings, span_ix),
                    "featured_span_embeddings": self._flatten_span_info(featured_span_embeddings, span_ix),
                    "span_entity_labels": self._flatten_span_info(span_entity_labels_one_hot, span_ix),
                    "span_features": self._flatten_span_info(span_features.float(), span_ix),
                    "valid": True,
                }

        return output_span_embeddings

    def coref_forward(self, output_span_embeddings, metadata):
        output_coref = {"loss": 0.0}
        if False and output_span_embeddings["valid"]:
            output_coref = self._coref.compute_representations(
                output_span_embeddings["spans"],
                output_span_embeddings["span_mask"],
                output_span_embeddings["link_embeddings"],
                metadata,
            )

        return output_coref

    def link_and_relation_forward(
        self, output_span_embedding, metadata, span_link_labels, relation_index, span_coref_labels, link_threshold=None
    ):
        output_relation = {"loss": 0.0, "metadata": metadata}
        output_linker = {"loss": 0.0}
        output_n_ary_relation = {"loss": 0.0}

        if output_span_embedding["valid"]:
            spans, featured_span_embeddings, span_ix = (
                output_span_embedding["spans"],
                output_span_embedding["featured_span_embeddings"],
                output_span_embedding["span_ix"],
            )
            # Linking

            span_link_labels = self._flatten_span_info(span_link_labels.unsqueeze(-1), span_ix)
            span_coref_labels = self._flatten_span_info(span_coref_labels, span_ix)
            output_linker = self._link_classifier(
                spans=spans,
                span_embeddings=featured_span_embeddings,
                span_features=output_span_embedding["span_features"],
                span_labels=span_link_labels.squeeze(-1),
                metadata=metadata,
            )

            if link_threshold is not None:
                # Keep only clusters with linked entities
                ner_probs = output_linker["ner_probs"].squeeze(0)
                ner_probs = (ner_probs > link_threshold).long()

                clusters_to_keep = (span_coref_labels.squeeze(0) * ner_probs[:, None]).sum(0) == 0
                span_coref_labels[:, :, clusters_to_keep] = 0

            if relation_index is not None or self.prediction_mode:
                # Relation Extraction
                output_relation = self._span_binary_relation.compute_representations(
                    spans, span_mask, link_embeddings, span_coref_labels, relation_index, metadata
                )

                cluster_type_batched = metadata[0]["document_metadata"]["type_to_clusters_map"]
                relation_index_dict = metadata[0]["document_metadata"]["relations_indexed"]

                output_n_ary_relation = self._cluster_n_ary_relation.compute_representations(
                    spans,
                    featured_span_embeddings,
                    span_coref_labels,
                    cluster_type_batched,
                    relation_index_dict,
                    metadata,
                )

        return output_relation, output_linker, output_n_ary_relation

    def ner_forward(self, output_embedding, ner_entity_labels, metadata):
        output_ner = {"loss": 0.0}

        output_ner = self._ner(
            output_embedding["contextualised"], output_embedding["mask"], ner_entity_labels, metadata
        )

        if self.prediction_mode:
            output_ner = self._ner.decode(output_ner)
            output_ner["spans"] = output_ner["spans"].to(output_embedding["text"].device).long()
            output_ner["span_labels"] = output_ner["span_labels"].to(output_embedding["text"].device).long()

        return output_ner

    def get_span_one_hot_labels(self, label_namespace, span_labels, spans):
        n_labels = self.vocab.get_vocab_size(label_namespace)
        span_labels_one_hot = torch.zeros((span_labels.size(0), span_labels.size(1), n_labels)).to(spans.device)
        span_labels_one_hot.scatter_(-1, span_labels.unsqueeze(-1), 1)
        return span_labels_one_hot

    @staticmethod
    def _flatten_span_info(span_info_batched, span_ix):
        feature_size = span_info_batched.size(-1)
        emb_flat = span_info_batched.view(-1, feature_size)
        span_info_flat = emb_flat[span_ix].unsqueeze(0)
        return span_info_flat

    @staticmethod
    def get_span_position(metadata, span_offset):
        doc_length = metadata[0]["document_metadata"]["doc_length"]
        span_position = span_offset.float().mean(-1, keepdim=True) / doc_length
        return span_position

    @staticmethod
    def offset_span_by_para_start(metadata, spans, span_mask):
        start_pos_in_doc = torch.LongTensor([x["start_pos_in_doc"] for x in metadata]).to(spans.device) #(B,)
        para_offset = start_pos_in_doc.unsqueeze(1).unsqueeze(2) #(B, 1, 1)
        span_offset = spans + (para_offset * span_mask.unsqueeze(-1).long())
        return span_offset

    def extract_span_embeddings(self, contextualized_embeddings, spans):
        attended_span_embeddings = self._attentive_span_extractor(contextualized_embeddings, spans)
        span_mask = (spans[:, :, 0] >= 0).long()
        spans = F.relu(spans.float()).long()
        endpoint_span_embeddings = self._endpoint_span_extractor(contextualized_embeddings, spans)
        span_embeddings = torch.cat([endpoint_span_embeddings, attended_span_embeddings], -1)
        return span_mask, spans, span_embeddings

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):
        res = {}
        res["ner"] = self._ner.decode(output_dict["ner"])
        res["relation"] = self._relation.decode(output_dict["relation"])
        res["linked"] = self._link_classifier.decode(output_dict["linked"])
        res["n_ary_relation"] = self._n_ary_relation.decode(output_dict["n_ary_relation"])

        return res

    def decode_relations(self, batch, link_threshold):
        output_embedding = self.embedding_forward(text=batch["text"])
        output_span_embedding = self.span_embeddings_forward(
            output_embedding=output_embedding,
            spans=batch["spans"],
            span_entity_labels=batch["span_entity_labels"],
            span_features=batch["span_features"],
            metadata=batch["metadata"],
        )

        output_relation, output_linker, output_n_ary_relation = self.link_and_relation_forward(
            output_span_embedding=output_span_embedding,
            metadata=batch["metadata"],
            span_link_labels=batch["span_link_labels"],
            relation_index=batch.get("relation_index", None),
            span_coref_labels=batch["span_coref_labels"],
            link_threshold=link_threshold,
        )

        res = {}
        res["relation"] = self._span_binary_relation.decode(output_relation)
        res["linked"] = self._link_classifier.decode(output_linker)
        res["n_ary_relation"] = self._cluster_n_ary_relation.decode(output_n_ary_relation)

        return res

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        Get all metrics from all modules. For the ones that shouldn't be displayed, prefix their
        keys with an underscore.
        """
        metrics_ner = self._ner.get_metrics(reset=reset)
        metrics_relation = self._span_binary_relation.get_metrics(reset=reset)
        metrics_link = self._link_classifier.get_metrics(reset=reset)
        metrics_n_ary = self._cluster_n_ary_relation.get_metrics(reset=reset)
        metrics_coref = self._coref.get_metrics(reset=reset)
        metrics_loss = {"loss_" + k: v.get_metric(reset) for k, v in self._multi_task_loss_metrics.items()}
        metrics_loss = {k: (v.item() if hasattr(v, "item") else v) for k, v in metrics_loss.items()}

        # Make sure that there aren't any conflicting names.
        metric_names = (
            list(metrics_ner.keys())
            + list(metrics_relation.keys())
            + list(metrics_link.keys())
            + list(metrics_n_ary.keys())
            + list(metrics_loss.keys())
            + list(metrics_coref.keys())
        )
        assert len(set(metric_names)) == len(metric_names)
        all_metrics = dict(
            list(metrics_ner.items())
            + list(metrics_relation.items())
            + list(metrics_link.items())
            + list(metrics_n_ary.items())
            + list(metrics_loss.items())
            + list(metrics_coref.items())
        )

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
