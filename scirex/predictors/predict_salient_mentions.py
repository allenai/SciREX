#! /usr/bin/env python

import json
import os
from sys import argv
from typing import Dict, List

from tqdm import tqdm

from allennlp.common.util import import_submodules
from allennlp.data import DataIterator, DatasetReader
from allennlp.data.dataset import Batch
from allennlp.models.archival import load_archive
from allennlp.nn import util as nn_util

import logging
logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s", level=logging.INFO)


def predict(archive_folder, test_file, output_file, cuda_device):
    '''
    test_file contains atleast - doc_id, sections, sentences, ner in scirex format.

    output_file - {
        'doc_id' : str,
        'saliency' : Tuple[start_index, end_index, salient (binary), saliency probability]
    }
    '''
    import_submodules("scirex")
    logging.info("Loading Model from %s", archive_folder)
    archive_file = os.path.join(archive_folder, "model.tar.gz")
    archive = load_archive(archive_file, cuda_device)
    model = archive.model
    model.eval()

    saliency_threshold = json.load(open(archive_folder + '/metrics.json'))['best_validation__span_threshold']

    model.prediction_mode = True
    config = archive.config.duplicate()
    dataset_reader_params = config["dataset_reader"]
    dataset_reader = DatasetReader.from_params(dataset_reader_params)
    dataset_reader.prediction_mode = True
    instances = dataset_reader.read(test_file)

    for instance in instances :
        batch = Batch([instance])
        batch.index_instances(model.vocab)

    data_iterator = DataIterator.from_params(config["validation_iterator"])
    iterator = data_iterator(instances, num_epochs=1, shuffle=False)

    with open(output_file, "w") as f:
        documents = {}
        for batch in tqdm(iterator):
            batch = nn_util.move_to_device(batch, cuda_device)  # Put on GPU.
            output_res = model.decode_saliency(batch, saliency_threshold)

            metadata = output_res['metadata']
            doc_ids: List[str] = [m["doc_id"] for m in metadata]
            assert len(set(doc_ids)) == 1

            decoded_spans: List[Dict[tuple, float]] = output_res['decoded_spans']
            doc_id = metadata[0]['doc_id']

            if doc_id not in documents :
                documents[doc_id] = {}
                documents[doc_id]['saliency'] = []
                documents[doc_id]['doc_id'] = doc_id

            for pspans in decoded_spans :
                for span, prob in pspans.items() :
                    documents[doc_id]['saliency'].append([span[0], span[1], 1 if prob > saliency_threshold else 0, prob])

        f.write("\n".join([json.dumps(x) for x in documents.values()]))


def main():
    archive_folder = argv[1]
    test_file = argv[2]
    output_file = argv[3]
    cuda_device = int(argv[4])
    predict(archive_folder, test_file, output_file, cuda_device)


if __name__ == "__main__":
    main()
