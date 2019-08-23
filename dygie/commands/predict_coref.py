#! /usr/bin/env python

import json
import os
from sys import argv
from typing import Dict, List, Tuple

from tqdm import tqdm

from allennlp.common.util import import_submodules
from allennlp.data import DataIterator
from allennlp.data.dataset import Batch
from allennlp.models.archival import load_archive
from allennlp.nn import util as nn_util
from dygie.data.dataset_readers.entity_linking_reader_all_pairs import \
    PwCLinkerAllPairsReader

def predict(archive_folder, test_file, output_file, cuda_device):
    import_submodules("dygie")
    archive_file = os.path.join(archive_folder, 'model.tar.gz')
    archive = load_archive(archive_file, cuda_device)
    model = archive.model
    model.eval()
    config = archive.config.duplicate()
    dataset_reader_params = config["dataset_reader"]
    dataset_reader = PwCLinkerAllPairsReader.from_params(params=dataset_reader_params)
    instances = dataset_reader.read(test_file)

    batch = Batch(instances)
    batch.index_instances(model.vocab)

    data_iterator = DataIterator.from_params(config["iterator"])
    iterator = data_iterator(instances, num_epochs=1, shuffle=False)

    with open(output_file, "w") as f:
        documents = {}
        for batch in tqdm(iterator):
            batch = nn_util.move_to_device(batch, cuda_device)  # Put on GPU.
            pred = model(**batch)
            decoded = model.decode(pred)

            metadata = decoded["metadata"]
            label_prob: List[float] = [float(x) for x in decoded["label_probs"]]
            doc_ids: List[str] = [m["doc_id"] for m in metadata]
            span_premise = [m['span_premise'] for m in metadata]
            span_hypothesis = [m['span_hypothesis'] for m in metadata]
            fields = [m['field'] for m in metadata]

            for doc_id, span_p, span_h, p, field in zip(doc_ids, span_premise, span_hypothesis, label_prob, fields):
                if doc_id not in documents:
                    documents[doc_id] = {"doc_id" : doc_id, "coref_prediction" : [], "coref_gold" : []}

                documents[doc_id]['coref_' + field].append(((span_p[0], span_p[1]), (span_h[0], span_h[1]), round(p, 3)))

        f.write("\n".join([json.dumps(x) for x in documents.values()]))


def main():
    archive_folder = argv[1]
    test_file = argv[2]
    output_file = argv[3]
    cuda_device = int(argv[4])
    predict(archive_folder, test_file, output_file, cuda_device)


if __name__ == "__main__":
    main()
