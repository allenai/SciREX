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

        for maybe_shuffled_instances in maybe_shuffled_docs:
            start = 0
            for batch_instances in lazy_groups_of(iter(maybe_shuffled_instances), self._batch_size):
                batch_instances = self._order_instances(batch_instances)
                num_e = sum([x.fields['metadata'].metadata['num_spans'] for x in batch_instances])
                batch_instances[0].fields['metadata'].metadata['span_idx'] = (start, start + num_e)
                start += num_e
                yield Batch(batch_instances)

    @staticmethod
    def _shuffle_documents(instances, shuffle: bool):
        """
        Randomly permute the documents for each batch
        """
        doc_ids = BatchIterator.unique(np.array([instance["metadata"]["doc_id"] for instance in instances]))
        if shuffle :
            doc_ids = np.random.permutation(doc_ids)
        res = []

        for doc in doc_ids:
            doc_instances = [ins for ins in instances if ins["metadata"]["doc_id"] == doc]
            paragraph_nums = [entry["metadata"]["paragraph_num"] for entry in doc_instances]
            assert paragraph_nums == list(range(len(doc_instances))), breakpoint()  # Make sure sentences are in order.
            res.append(doc_instances)
        assert len([x for y in res for x in y]) == len(instances)
        return res

    @staticmethod
    def _order_instances(instances):
        instances = sorted(instances, key=lambda x: x["metadata"]["paragraph_num"])
        return instances

    def get_num_batches(self, instances) -> int:
        doc_ids = np.unique(np.array([instance["metadata"]["doc_id"] for instance in instances]))
        n_batches = [
            math.ceil(len([ins for ins in instances if ins["metadata"]["doc_id"] == doc]) / self._batch_size)
            for doc in doc_ids
        ]

        logging.info(pd.Series(n_batches).describe().to_string())
        return sum(n_batches)

    @staticmethod
    def unique(array):
        uniq, index = np.unique(array, return_index=True)
        return uniq[index.argsort()]
