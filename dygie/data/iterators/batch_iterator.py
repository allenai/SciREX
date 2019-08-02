from collections import deque
from typing import Iterable, Deque
import logging
import numpy as np
from random import shuffle as random_shuffle

from overrides import overrides

from allennlp.common.util import lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.dataset import Batch

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
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        # Shuffle the documents if requested.
        maybe_shuffled_docs = self._shuffle_documents(instances, shuffle)
        
        class_weight = self.generate_class_weight(maybe_shuffled_docs, 'ner_labels')
        class_weight_entity = self.generate_class_weight(maybe_shuffled_docs, 'ner_entity_labels')
        class_weight_link = self.generate_class_weight(maybe_shuffled_docs, 'ner_link_labels')
        print(class_weight)
        print(class_weight_entity)
        print(class_weight_link)

        for doc in maybe_shuffled_docs :
            for ins in doc :
                ins.fields['metadata'].metadata['ner_labels_class_weight'] = class_weight
                ins.fields['metadata'].metadata['ner_entity_labels_class_weight'] = class_weight_entity
                ins.fields['metadata'].metadata['ner_link_labels_class_weight'] = class_weight_link

        for maybe_shuffled_instances in maybe_shuffled_docs :
            for instance_list in self._memory_sized_lists(maybe_shuffled_instances):
                iterator = iter(instance_list)
                excess: Deque[Instance] = deque()
                # Then break each memory-sized list into batches.
                for batch_instances in lazy_groups_of(iterator, self._batch_size):
                    for possibly_smaller_batches in self._ensure_batch_is_sufficiently_small(batch_instances, excess):
                        batch = Batch(self.order_instances(possibly_smaller_batches))
                        yield batch
                if excess:
                    yield Batch(self.order_instances(excess))

    def generate_class_weight(self, docs, label_field):
        labels = [label for doc in docs for ins in doc for label in ins.fields[label_field].labels]
        label_set = sorted(list(set(labels)))
        class_weight = compute_class_weight('balanced', label_set, labels)
        class_weight = {k:v for k, v in zip(label_set, class_weight)}
        return class_weight

    @staticmethod
    def _shuffle_documents(instances, shuffle:bool, shuffle_instances:bool):
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
            if shuffle_instances :
                random_shuffle(doc_instances)
            res.append(doc_instances)
        assert len([x for y in res for x in y]) == len(instances)
        return res

    @staticmethod
    def order_instances(instances) :
        instances = sorted(instances, key=lambda x : x["metadata"]["sentence_num"])
        return instances
