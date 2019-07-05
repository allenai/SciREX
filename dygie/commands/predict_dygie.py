#! /usr/bin/env python

"""
Make predictions of trained model, output as json like input. Not easy to do this in the current
AllenNLP predictor framework, so here's a short script to do it.

usage: predict.py [archive-file] [test-file] [output-file]
"""

# TODO(dwadden) This breaks right now on relation prediction because json can't do dicts whose keys
# are tuples.

import json
from sys import argv

import numpy as np

from allennlp.models.archival import load_archive
from allennlp.common.util import import_submodules
from allennlp.data import DatasetReader
from allennlp.data.dataset import Batch
from allennlp.nn import util as nn_util

from dygie.data.iterators.document_iterator import DocumentIterator


decode_fields = dict(coref="clusters",
                     ner="decoded_ner",
                     relation="decoded_relations")

decode_names = dict(coref="predicted_clusters",
                    ner="predicted_ner",
                    relation="predicted_relations")


def cleanup(k, decoded, sentence_starts):
    dispatch = {"coref": cleanup_coref,
                "ner": cleanup_ner,
                "relation": cleanup_relation}
    return dispatch[k](decoded, sentence_starts)


def cleanup_coref(decoded, sentence_starts):
    "Convert from nested list of tuples to nested list of lists."
    # The coref code assumes batch sizes other than 1. We only have 1.
    assert len(decoded) == 1
    decoded = decoded[0]
    res = []
    for cluster in decoded:
        cleaned = [list(x) for x in cluster]  # Convert from tuple to list.
        res.append(cleaned)
    return res


def cleanup_ner(decoded, sentence_starts):
    assert len(decoded) == len(sentence_starts)
    res = []
    for sentence, sentence_start in zip(decoded, sentence_starts):
        res_sentence = []
        for tag in sentence:
            new_tag = [tag[0] + sentence_start, tag[1] + sentence_start, tag[2]]
            res_sentence.append(new_tag)
        res.append(res_sentence)
    return res


def cleanup_relation(decoded, sentence_starts):
    "Add sentence offsets to relation results."
    assert len(decoded) == len(sentence_starts)  # Length check.
    res = []
    for sentence, sentence_start in zip(decoded, sentence_starts):
        res_sentence = []
        for rel in sentence:
            cleaned = [x + sentence_start for x in rel[:4]] + [rel[4]]
            res_sentence.append(cleaned)
        res.append(res_sentence)
    return res


def load_json(test_file):
    res = []
    with open(test_file, "r") as f:
        for line in f:
            res.append(json.loads(line))

    return res


def check_lengths(d):
    "Make sure all entries in dict have same length."
    keys = list(d.keys())
    keys.remove("doc_key")
    lengths = [len(d[k]) for k in keys]
    assert len(set(lengths)) == 1


def predict(archive_file, test_file, output_file, cuda_device):
    import_submodules("dygie")
    gold_test_data = load_json(test_file)
    archive = load_archive(archive_file, cuda_device)
    model = archive.model
    model.eval()
    config = archive.config.duplicate()
    dataset_reader_params = config["dataset_reader"]
    dataset_reader = DatasetReader.from_params(dataset_reader_params)
    instances = dataset_reader.read(test_file)
    batch = Batch(instances)
    batch.index_instances(model.vocab)
    iterator = DocumentIterator()
    with open(output_file, "w") as f:
        for doc, gold_data in zip(iterator(batch.instances, num_epochs=1, shuffle=False),
                                  gold_test_data):
            doc = nn_util.move_to_device(doc, cuda_device)  # Put on GPU.
            sentence_lengths = [len(entry["sentence"]) for entry in doc["metadata"]]
            sentence_starts = np.cumsum(sentence_lengths)
            sentence_starts = np.roll(sentence_starts, 1)
            sentence_starts[0] = 0
            pred = model(**doc)
            decoded = model.decode(pred)
            predictions = {}
            for k, v in decoded.items():
                predictions[decode_names[k]] = cleanup(k, v[decode_fields[k]], sentence_starts)
            res = {}
            res.update(gold_data)
            res.update(predictions)
            check_lengths(res)
            encoded = json.dumps(res, default=int)
            f.write(encoded + "\n")


def main():
    archive_file = argv[1]
    test_file = argv[2]
    output_file = argv[3]
    cuda_device = int(argv[4])
    predict(archive_file, test_file, output_file, cuda_device)


if __name__ == '__main__':
    main()
