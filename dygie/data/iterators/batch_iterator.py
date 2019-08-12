from collections import deque
from typing import Iterable, Deque, Tuple
import logging
import numpy as np
from random import shuffle as random_shuffle

from allennlp.common.util import lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.dataset import Batch

import torch
from collections import Counter

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
from sklearn.utils.class_weight import compute_class_weight


@DataIterator.register("ie_batch")
class BatchIterator(DataIterator):
    """
    For multi-task IE, we want the training instances in a batch to be successive sentences from the
    same document. Otherwise the coreference labels don't make sense.

    At train time, if `shuffle` is True, shuffle the documents but not the instances within them.
    Then, do the same thing as AllenNLP BasicIterator.
    """

    def __init__(
        self,
        shuffle_instances: bool = False,
        batch_size: int = 32,
        instances_per_epoch: int = None,
        max_instances_in_memory: int = None,
        cache_instances: bool = False,
        track_epoch: bool = False,
        maximum_samples_per_batch: Tuple[str, int] = None,
    ):
        super().__init__(
            cache_instances=cache_instances,
            track_epoch=track_epoch,
            batch_size=batch_size,
            instances_per_epoch=instances_per_epoch,
            max_instances_in_memory=max_instances_in_memory,
            maximum_samples_per_batch=maximum_samples_per_batch,
        )
        self._shuffle_instances = shuffle_instances

    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        maybe_shuffled_docs = self._shuffle_documents(instances, shuffle, self._shuffle_instances)

        class_weights = {}
        sample_prob = {}
        for field in ["ner_labels"]:
            class_weights[field], sample_prob[field] = self.generate_class_weight(maybe_shuffled_docs, field)

        for doc in maybe_shuffled_docs:
            for ins in doc:
                for field in ["ner_labels"]:
                    ins.fields["metadata"].metadata[field + "_class_weight"] = class_weights[field]
                    ins.fields["metadata"].metadata[field + "_sample_prob"] = sample_prob[field]

        for maybe_shuffled_instances in maybe_shuffled_docs:
            for instance_list in self._memory_sized_lists(maybe_shuffled_instances):
                iterator = iter(instance_list)
                excess: Deque[Instance] = deque()
                for batch_instances in lazy_groups_of(iterator, self._batch_size):
                    for possibly_smaller_batches in self._ensure_batch_is_sufficiently_small(batch_instances, excess):
                        batch = Batch(self.order_instances(possibly_smaller_batches))
                        yield batch
                if excess:
                    yield Batch(self.order_instances(excess))

    def generate_class_weight(self, docs, label_field):
        labels = [label for doc in docs for ins in doc for label in ins.fields[label_field].labels]
        label_set = sorted(list(set(labels)))
        class_weight = compute_class_weight("balanced", label_set, labels)
        class_weight = {k: v**(1/4) for k, v in zip(label_set, class_weight)}

        c = dict(Counter([l for l in labels if l != '']))
        count_non_null = [v for k, v in c.items() if k != '']
        per90 = float(np.percentile(count_non_null, 90))
        prob = {k:(min(1, per90/v if k != '' else 1.0)) for k, v in c.items()}

        return class_weight, prob

    @staticmethod
    def _shuffle_documents(instances, shuffle: bool, shuffle_instances: bool):
        """
        Randomly permute the documents for each batch
        """
        doc_keys = np.array([instance["metadata"]["doc_key"] for instance in instances])
        shuffled = np.random.permutation(np.unique(doc_keys)) if shuffle else np.unique(doc_keys)
        res = []
        for doc in shuffled:
            ixs = np.nonzero(doc_keys == doc)[0].tolist()
            doc_instances = [instances[ix] for ix in ixs]
            sentence_nums = [entry["metadata"]["sentence_num"] for entry in doc_instances]
            assert sentence_nums == list(range(len(doc_instances)))  # Make sure sentences are in order.
            if shuffle_instances:
                random_shuffle(doc_instances)
            res.append(doc_instances)
        assert len([x for y in res for x in y]) == len(instances)
        return res

    @staticmethod
    def order_instances(instances):
        instances = sorted(instances, key=lambda x: x["metadata"]["sentence_num"])
        return instances
