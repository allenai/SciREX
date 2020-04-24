#! /usr/bin/env python

import json
import os
from sys import argv
from typing import Dict, List, Tuple

from tqdm import tqdm

from allennlp.common.util import import_submodules
from allennlp.data import DataIterator, DatasetReader
from allennlp.data.dataset import Batch
from allennlp.models.archival import load_archive
from allennlp.nn import util as nn_util

from scirex_utilities.json_utilities import NumpyEncoder

import logging

logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s", level=logging.INFO)


def predict(archive_folder, test_file, output_file, cuda_device):
    import_submodules("scirex")
    logging.info("Loading Model from %s", archive_folder)
    archive_file = os.path.join(archive_folder, "model.tar.gz")
    archive = load_archive(archive_file, cuda_device)
    model = archive.model
    model.eval()

    model.prediction_mode = True
    config = archive.config.duplicate()
    dataset_reader_params = config["dataset_reader"]
    dataset_reader = DatasetReader.from_params(dataset_reader_params)
    instances = dataset_reader.read(test_file)

    for instance in instances:
        batch = Batch([instance])
        batch.index_instances(model.vocab)

    data_iterator = DataIterator.from_params(config["validation_iterator"])
    iterator = data_iterator(instances, num_epochs=1, shuffle=False)

    with open(output_file, "w") as f:
        documents = {}
        for batch in tqdm(iterator):
            batch = nn_util.move_to_device(batch, cuda_device)  # Put on GPU.

            output_embedding = model.embedding_forward(batch["text"])
            output_ner = model.ner_forward(output_embedding, batch["ner_type_labels"], batch["metadata"])
            predicted_ner: List[Dict[Tuple[int, int], str]] = output_ner["decoded_ner"]

            metadata = output_ner["metadata"]
            doc_ids: List[str] = [m["doc_id"] for m in metadata]
            assert len(set(doc_ids)) == 1
            para_ids: List[int] = [m["paragraph_num"] for m in metadata]
            para_starts: List[int] = [int(m["start_pos_in_doc"]) for m in metadata]
            para_ends: List[int] = [int(m["end_pos_in_doc"]) for m in metadata]
            sentence_indices: List[List[Tuple[int, int]]] = [m["sentence_indices"] for m in metadata]
            words: List[str] = [m["paragraph"] for m in metadata]

            for s, e, sents in zip(para_starts, para_ends, sentence_indices):
                assert s == sents[0][0], breakpoint()
                assert e == sents[-1][-1], breakpoint()

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
                res["sentence_indices"] = sentence_indices[i]
                res["prediction"] = [(k[0], k[1], v) for k, v in predicted_ner[i].items()]
                documents[doc_ids[i]].append(res)

        documents = process_documents(documents)

        f.write("\n".join([json.dumps(x, cls=NumpyEncoder) for x in documents.values()]))


def process_documents(documents):
    for k, v in documents.items():
        v = sorted(v, key=lambda x: x["para_start"])
        for p, q in zip(v[:-1], v[1:]):
            if p["para_end"] != q["para_start"]:
                breakpoint()

        words = [w for x in v for w in x["words"]]
        paragraphs = [[x["para_start"], x["para_end"]] for x in v]
        paragraph_sentences = [s for x in v for s in x["sentence_indices"]]
        predictions = [
            (s + x["para_start"], e + x["para_start"], l)
            for x in v
            for (s, e, l) in x["prediction"]
        ]

        documents[k] = {
            "words": words,
            "sections": paragraphs,
            "sentences": paragraph_sentences,
            "ner": predictions,
            "doc_id": k,
        }

    return documents


def main():
    archive_folder = argv[1]
    test_file = argv[2]
    output_file = argv[3]
    cuda_device = int(argv[4])
    predict(archive_folder, test_file, output_file, cuda_device)


if __name__ == "__main__":
    main()
