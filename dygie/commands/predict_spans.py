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

from typing import List, Dict, Tuple
import numpy as np

from allennlp.models.archival import load_archive
from allennlp.common.util import import_submodules
from allennlp.data import DatasetReader
from allennlp.data import DataIterator
from allennlp.data.dataset import Batch
from allennlp.nn import util as nn_util

from dygie.data.iterators.batch_iterator import BatchIterator


def load_json(test_file):
    res = []
    with open(test_file, "r") as f:
        for line in f:
            res.append(json.loads(line))

    return res


def predict(archive_file, test_file, output_file, cuda_device):
    import_submodules("dygie")
    archive = load_archive(archive_file, cuda_device)
    model = archive.model
    model.eval()
    config = archive.config.duplicate()
    dataset_reader_params = config["dataset_reader"]
    dataset_reader = DatasetReader.from_params(dataset_reader_params)
    instances = dataset_reader.read(test_file)

    batch = Batch(instances)
    batch.index_instances(model.vocab)

    data_iterator = DataIterator.from_params(config["validation_iterator"])
    iterator = data_iterator(instances, num_epochs=1, shuffle=False)

    with open(output_file, "w") as f:
        documents = {}
        for batch in iterator:
            batch = nn_util.move_to_device(batch, cuda_device)  # Put on GPU.
            pred = model(**batch)
            decoded = model.decode(pred)
            predicted_ner: List[Dict[Tuple[int, int], str]] = decoded["ner"]["decoded_ner"]
            gold_ner: List[Dict[Tuple[int, int], str]] = decoded["ner"]["gold_ner"]

            metadata = decoded["ner"]["metadata"]
            doc_ids: List[str] = [m["doc_key"] for m in metadata]
            para_ids: List[int] = [m["sentence_num"] for m in metadata]
            para_starts: List[int] = [int(m["start_pos_in_doc"]) for m in metadata]
            para_ends: List[int] = [int(m["end_pos_in_doc"]) for m in metadata]
            words: List[str] = [m["sentence"] for m in metadata]

            assert len(set(doc_ids)) == 1
            for i in range(len(para_ids)):
                res = {}
                if doc_ids[i] not in documents:
                    documents[doc_ids[i]] = []

                res["doc_id"] = doc_ids[i]
                res["para_id"] = para_ids[i]
                res["para_start"] = para_starts[i]
                res["para_end"] = para_ends[i]
                res["words"] = words[i]
                res["prediction"] = [{"span": k, "label": v} for k, v in predicted_ner[i].items()]
                res["gold"] = [{"span": k, "label": v} for k, v in gold_ner[i].items()]
                documents[doc_ids[i]].append(res)

        documents = process_documents(documents)
        f.write("\n".join([json.dumps(x) for x in documents.values()]))


def process_documents(documents):
    for k, v in documents.items():
        v = sorted(v, key=lambda x: x["para_start"])
        for p, q in zip(v[:-1], v[1:]):
            if p["para_end"] != q["para_start"]:
                breakpoint()

        words = [w for x in v for w in x["words"]]
        paragraphs = [[x["para_start"], x["para_end"]] for x in v]
        predictions = [
            {"span": (e["span"][0] + x["para_start"], e["span"][1] + x["para_start"]), "label": e["label"]}
            for x in v
            for e in x["prediction"]
        ]

        golds = [
            {"span": (e["span"][0] + x["para_start"], e["span"][1] + x["para_start"]), "label": e["label"]}
            for x in v
            for e in x["gold"]
        ]

        documents[k] = {
            'words' : words, 'paragraphs' : paragraphs, 'prediction' : predictions, 'gold' : golds, 'doc_id' : k
        }

    return documents


def main():
    archive_file = argv[1]
    test_file = argv[2]
    output_file = argv[3]
    cuda_device = int(argv[4])
    predict(archive_file, test_file, output_file, cuda_device)


if __name__ == "__main__":
    main()
