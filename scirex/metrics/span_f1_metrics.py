from typing import Dict, List, Optional, Set, Callable
from collections import defaultdict

import torch

from allennlp.common.checks import ConfigurationError
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics.metric import Metric
from allennlp.data.dataset_readers.dataset_utils.span_utils import (
        bio_tags_to_spans,
        bioul_tags_to_spans,
        iob1_tags_to_spans,
        bmes_tags_to_spans,
        TypedStringSpan
)


TAGS_TO_SPANS_FUNCTION_TYPE = Callable[[List[str], Optional[List[str]]], List[TypedStringSpan]]


class SpanBasedF1Measure(Metric):
    """
    Copied (Span Based F1 Measure from allennlp)

    """
    def __init__(self,
                 label_vocabulary: Dict[int, str],
                 ignore_classes: List[str] = None,
                 label_encoding: Optional[str] = "BIO") -> None:

        if label_encoding:
            if label_encoding not in ["BIO", "IOB1", "BIOUL", "BMES"]:
                raise ConfigurationError("Unknown label encoding - expected 'BIO', 'IOB1', 'BIOUL', 'BMES'.")

        self._label_encoding = label_encoding
        self._label_vocabulary = label_vocabulary
        self._ignore_classes: List[str] = ignore_classes or []
        # These will hold per label span counts.
        self._true_positives: Dict[str, int] = defaultdict(int)
        self._false_positives: Dict[str, int] = defaultdict(int)
        self._false_negatives: Dict[str, int] = defaultdict(int)

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None,
                 prediction_map: Optional[torch.Tensor] = None):
                 
        if mask is None:
            mask = torch.ones_like(gold_labels)

        predictions, gold_labels, mask, prediction_map = self.unwrap_to_tensors(predictions,
                                                                                gold_labels,
                                                                                mask, prediction_map)

        num_classes = predictions.size(-1)
        if (gold_labels >= num_classes).any():
            raise ConfigurationError("A gold label passed to SpanBasedF1Measure contains an "
                                     "id >= {}, the number of classes.".format(num_classes))

        sequence_lengths = get_lengths_from_binary_sequence_mask(mask)
        argmax_predictions = predictions.max(-1)[1]

        if prediction_map is not None:
            argmax_predictions = torch.gather(prediction_map, 1, argmax_predictions)
            gold_labels = torch.gather(prediction_map, 1, gold_labels.long())

        argmax_predictions = argmax_predictions.float()

        # Iterate over timesteps in batch.
        batch_size = gold_labels.size(0)
        for i in range(batch_size):
            sequence_prediction = argmax_predictions[i, :]
            sequence_gold_label = gold_labels[i, :]
            length = sequence_lengths[i]

            if length == 0:
                # It is possible to call this metric with sequences which are
                # completely padded. These contribute nothing, so we skip these rows.
                continue

            predicted_string_labels = [self._label_vocabulary[label_id]
                                       for label_id in sequence_prediction[:length].tolist()]
            gold_string_labels = [self._label_vocabulary[label_id]
                                  for label_id in sequence_gold_label[:length].tolist()]

            tags_to_spans_function = None
            if self._label_encoding == "BIO":
                tags_to_spans_function = bio_tags_to_spans
            elif self._label_encoding == "IOB1":
                tags_to_spans_function = iob1_tags_to_spans
            elif self._label_encoding == "BIOUL":
                tags_to_spans_function = bioul_tags_to_spans
            elif self._label_encoding == "BMES":
                tags_to_spans_function = bmes_tags_to_spans

            predicted_spans = tags_to_spans_function(predicted_string_labels, self._ignore_classes)
            gold_spans = tags_to_spans_function(gold_string_labels, self._ignore_classes)

            typed_predicted_spans = defaultdict(list)
            typed_gold_spans = defaultdict(list)

            for label, span in predicted_spans :
                typed_predicted_spans[label].append(span)

            for label, span in gold_spans :
                typed_gold_spans[label].append(span)

            for label in typed_predicted_spans :
                for p_span in typed_predicted_spans[label] :
                    matched = False
                    for g_span in typed_gold_spans[label] :
                        iou = span_match(p_span, g_span)
                        if iou > 0.5 :
                            self._true_positives[label] += 1
                            matched = True
                            typed_gold_spans[label].remove(g_span)
                            break
                    if not matched :
                        self._false_positives[label] += 1

            for label in typed_gold_spans :
                for span in typed_gold_spans[label] :
                    self._false_negatives[label] += 1

            # for span in predicted_spans:
            #     if span in gold_spans:
            #         self._true_positives[span[0]] += 1
            #         gold_spans.remove(span)
            #     else:
            #         self._false_positives[span[0]] += 1
            # # These spans weren't predicted.
            # for span in gold_spans:
            #     self._false_negatives[span[0]] += 1

    def get_metric(self, reset: bool = False):
        all_tags: Set[str] = set()
        all_tags.update(self._true_positives.keys())
        all_tags.update(self._false_positives.keys())
        all_tags.update(self._false_negatives.keys())
        all_metrics = {}
        for tag in all_tags:
            precision, recall, f1_measure = self._compute_metrics(self._true_positives[tag],
                                                                  self._false_positives[tag],
                                                                  self._false_negatives[tag])
            precision_key = "precision" + "-" + tag
            recall_key = "recall" + "-" + tag
            f1_key = "f1-measure" + "-" + tag
            all_metrics[precision_key] = precision
            all_metrics[recall_key] = recall
            all_metrics[f1_key] = f1_measure

        # Compute the precision, recall and f1 for all spans jointly.
        precision, recall, f1_measure = self._compute_metrics(sum(self._true_positives.values()),
                                                              sum(self._false_positives.values()),
                                                              sum(self._false_negatives.values()))
        all_metrics["precision-overall"] = precision
        all_metrics["recall-overall"] = recall
        all_metrics["f1-measure-overall"] = f1_measure
        if reset:
            self.reset()
        return all_metrics

    @staticmethod
    def _compute_metrics(true_positives: int, false_positives: int, false_negatives: int):
        precision = float(true_positives) * 100 / float(true_positives + false_positives + 1e-13)
        recall = float(true_positives) * 100 / float(true_positives + false_negatives + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
        return precision, recall, f1_measure

    def reset(self):
        self._true_positives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)

def span_match(span_1, span_2) :
    sa, ea = span_1
    sb, eb = span_2
    ea, eb = ea + 1, eb + 1
    iou = (min(ea, eb) - max(sa, sb)) / (max(eb, ea) - min(sa, sb))
    return iou