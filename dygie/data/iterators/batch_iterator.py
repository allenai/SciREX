from typing import Iterable
import logging
import numpy as np
import math

import pandas as pd

from allennlp.common.util import lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.dataset import Batch

from collections import Counter

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
from sklearn.utils.class_weight import compute_class_weight


@DataIterator.register("ie_batch")
class BatchIterator(DataIterator):
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        maybe_shuffled_docs = self._shuffle_documents(instances, shuffle)
        maybe_shuffled_docs = self._add_weights(maybe_shuffled_docs)

        for maybe_shuffled_instances in maybe_shuffled_docs:
            start = 0
            for batch_instances in lazy_groups_of(iter(maybe_shuffled_instances), self._batch_size):
                batch_instances = self._order_instances(batch_instances)
                num_e = sum([x.fields['metadata'].metadata['num_spans'] for x in batch_instances])
                batch_instances[0].fields['metadata'].metadata['span_idx'] = (start, start + num_e)
                start += num_e
                yield Batch(batch_instances)

    def _add_weights(self, maybe_shuffled_docs):
        class_weights = {}
        sample_prob = {}
        for field in ["span_entity_labels"]:
            class_weights[field], sample_prob[field] = self._generate_class_weight(maybe_shuffled_docs, field)

        for doc in maybe_shuffled_docs:
            for ins in doc:
                for field in ["span_entity_labels"]:
                    ins.fields["metadata"].metadata[field + "_class_weight"] = class_weights[field]
                    ins.fields["metadata"].metadata[field + "_sample_prob"] = sample_prob[field]

        return maybe_shuffled_docs

    def _generate_class_weight(self, docs, label_field):
        labels = [
            label
            for doc in docs
            for ins in doc
            if label_field in ins.fields
            for label in ins.fields[label_field].labels
        ]
        label_set = sorted(list(set(labels)))
        class_weight = compute_class_weight("balanced", label_set, labels)
        class_weight = {k: v if v > 1 else v ** (1 / 2) for k, v in zip(label_set, class_weight)}

        c = dict(Counter([l for l in labels if l != ""]))
        count_non_null = [v for k, v in c.items() if k != ""]
        per90 = float(np.percentile(count_non_null, 90))
        prob = {k: (min(1, per90 / v if k != "" else 1.0)) for k, v in c.items()}

        return class_weight, prob

    @staticmethod
    def _shuffle_documents(instances, shuffle: bool):
        """
        Randomly permute the documents for each batch
        """
        doc_keys = BatchIterator.unique(np.array([instance["metadata"]["doc_key"] for instance in instances]))
        if shuffle :
            doc_keys = np.random.permutation(doc_keys)
        res = []

        for doc in doc_keys:
            doc_instances = [ins for ins in instances if ins["metadata"]["doc_key"] == doc]
            sentence_nums = [entry["metadata"]["sentence_num"] for entry in doc_instances]
            assert sentence_nums == list(range(len(doc_instances))), breakpoint()  # Make sure sentences are in order.
            res.append(doc_instances)
        assert len([x for y in res for x in y]) == len(instances)
        return res

    @staticmethod
    def _order_instances(instances):
        instances = sorted(instances, key=lambda x: x["metadata"]["sentence_num"])
        return instances

    def get_num_batches(self, instances) -> int:
        doc_keys = np.unique(np.array([instance["metadata"]["doc_key"] for instance in instances]))
        n_batches = [
            math.ceil(len([ins for ins in instances if ins["metadata"]["doc_key"] == doc]) / self._batch_size)
            for doc in doc_keys
        ]

        logging.info(pd.Series(n_batches).describe().to_string())
        return sum(n_batches)

    @staticmethod
    def unique(array):
        uniq, index = np.unique(array, return_index=True)
        return uniq[index.argsort()]
