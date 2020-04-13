#! /usr/bin/env python

import json
import os
from sys import argv
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from allennlp.common.util import import_submodules
from allennlp.data import DataIterator, DatasetReader
from allennlp.data.dataset import Batch
from allennlp.models.archival import load_archive
from allennlp.nn import util as nn_util

import logging
logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s", level=logging.INFO)


def predict(archive_folder, test_file, output_file, cuda_device):
    link_threshold = json.load(open(archive_folder + '/metrics.json'))['best_validation__span_threshold']
    relation_threshold = json.load(open(archive_folder + '/metrics_test.json'))['_n_ary_rel_global_threshold']
    print(relation_threshold)
    
    import_submodules("dygie")
    logging.info("Loading Model from %s", archive_folder)
    archive_file = os.path.join(archive_folder, "model.tar.gz")
    archive = load_archive(archive_file, cuda_device)
    model = archive.model
    model.eval()

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
            for key in batch :
                try :
                    if key == 'text' :
                        for key_1 in batch[key] :
                            try :
                                batch[key][key_1] = nn_util.move_to_device(batch[key][key_1], cuda_device)
                            except :
                                print(key_1)
                                breakpoint()
                    else :
                        batch[key] = nn_util.move_to_device(batch[key], cuda_device)
                except :
                    print(key)
                    breakpoint()
            # batch =   # Put on GPU.
            output_res = model.decode_relations(batch, link_threshold)

            # linked_spans : Dict[(int, int), str] = output_res['linked']['decoded_spans']
            n_ary_relations = output_res['n_ary_relation']
            predicted_relations, scores = n_ary_relations['candidates'], n_ary_relations['scores']

            metadata = output_res['n_ary_relation']['metadata'][0]
            doc_id = metadata['doc_key']
            coref_key_map = {k:i for i, k in metadata['document_metadata']['cluster_name_to_id'].items()}
        
            for i, rel in enumerate(predicted_relations) :
                predicted_relations[i] = tuple([coref_key_map[k] if k in coref_key_map else None for k in rel])

            if doc_id not in documents :
                documents[doc_id] = {'predicted_relations' : [], 'scores' : [], 'doc_id' : doc_id, "linked_clusters" : [], 'is_true' :[]}

            documents[doc_id]['predicted_relations'] += predicted_relations
            documents[doc_id]['scores'] += [round(float(x), 4) for x in list(scores.ravel())]
            documents[doc_id]['is_true'] += [1 if x > relation_threshold else 0 for x in list(scores.ravel())]

        f.write("\n".join([json.dumps(x) for x in documents.values()]))


def main():
    archive_folder = argv[1]
    test_file = argv[2]
    output_file = argv[3]
    cuda_device = int(argv[4])
    predict(archive_folder, test_file, output_file, cuda_device)


if __name__ == "__main__":
    main()
