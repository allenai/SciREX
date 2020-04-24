#! /usr/bin/env python

import json
import os
from sys import argv
from typing import List

from tqdm import tqdm
import torch

from allennlp.common.util import import_submodules
from allennlp.data import DataIterator
from allennlp.data.dataset import Batch
from allennlp.models.archival import load_archive
from allennlp.nn import util as nn_util
from scirex.data.dataset_readers.coreference_eval_reader import ScirexCoreferenceEvalReader


def predict(archive_folder, span_prediction_file, output_file, cuda_device):
    '''
    span_prediction_file (jsonl) needs atleast three fields 
        - doc_id, words: List[str], field: List[Tuple[start_index, end_index, type]]

    Return output_file (jsonl) - 
        {
            'doc_id' : str,
            'pairwise_coreference_scores' : List[(s_1, e_1), (s_2, e_2), float (3 sig. digits) in [0, 1]]
        }
    '''
    import_submodules("scirex")
    archive_file = os.path.join(archive_folder, "model.tar.gz")
    archive = load_archive(archive_file, cuda_device)
    model = archive.model
    model.eval()
    config = archive.config.duplicate()
    dataset_reader_params = config["dataset_reader"]
    dataset_reader_params.pop('type')
    dataset_reader = ScirexCoreferenceEvalReader.from_params(params=dataset_reader_params, field="ner")
    instances = dataset_reader.read(span_prediction_file)

    batch = Batch(instances)
    batch.index_instances(model.vocab)

    config['iterator'].pop('batch_size')
    data_iterator = DataIterator.from_params(config["iterator"], batch_size=1000)
    iterator = data_iterator(instances, num_epochs=1, shuffle=False)

    with open(output_file, "w") as f:
        documents = {}
        for batch in tqdm(iterator):
            with torch.no_grad() :
                batch = nn_util.move_to_device(batch, cuda_device)  # Put on GPU.
                pred = model(**batch)
                decoded = model.decode(pred)

            metadata = decoded["metadata"]
            label_prob: List[float] = [float(x) for x in decoded["label_probs"]]
            doc_ids: List[str] = [m["doc_id"] for m in metadata]
            span_premise = [m["span_premise"] for m in metadata]
            span_hypothesis = [m["span_hypothesis"] for m in metadata]
            fields = [m["field"] for m in metadata]
            assert len(set(fields)) == 1, breakpoint()

            for doc_id, span_p, span_h, p in zip(doc_ids, span_premise, span_hypothesis, label_prob):
                if doc_id not in documents:
                    documents[doc_id] = {"doc_id": doc_id, "pairwise_coreference_scores": []}

                documents[doc_id]["pairwise_coreference_scores"].append(
                    ((span_p[0], span_p[1]), (span_h[0], span_h[1]), round(p, 4))
                )

        f.write("\n".join([json.dumps(x) for x in documents.values()]))


def main():
    archive_folder = argv[1]
    test_file = argv[2]
    output_file = argv[3]
    cuda_device = int(argv[4])
    predict(archive_folder, test_file, output_file, cuda_device)


if __name__ == "__main__":
    main()
