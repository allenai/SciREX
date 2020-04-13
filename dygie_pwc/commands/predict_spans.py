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
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)


def predict(archive_folder, test_file, output_file, cuda_device):
    import_submodules("dygie")
    logging.info("Loading Model from %s", archive_folder)
    archive_file = os.path.join(archive_folder, "model.tar.gz")
    linking_threshold = json.load(open(archive_folder + '/metrics.json'))['best_validation__span_threshold']
    archive = load_archive(archive_file, cuda_device)
    model = archive.model
    model.eval()

    model.prediction_mode = True
    config = archive.config.duplicate()
    dataset_reader_params = config["dataset_reader"]
    dataset_reader = DatasetReader.from_params(dataset_reader_params)
    instances = dataset_reader.read(test_file)

    batch = Batch(instances)
    batch.index_instances(model.vocab)
    
    config['validation_iterator']['batch_size'] = 10

    data_iterator = DataIterator.from_params(config["validation_iterator"])
    iterator = data_iterator(instances, num_epochs=1, shuffle=False)

    with open(output_file, "w") as f:
        documents = {}
        documents_relations = {}
        for batch in tqdm(iterator):
            batch = nn_util.move_to_device(batch, cuda_device)  # Put on GPU.
            batch["spans"] = None
            batch["span_coref_labels"] = None
            batch["relation_index"] = None
            batch['span_link_labels'] = None
            batch['span_entity_labels'] = None
            
            pred = model(**batch)
            decoded = model.decode(pred)
            predicted_ner: List[Dict[Tuple[int, int], str]] = decoded["ner"]["decoded_ner"]
            gold_ner: List[Dict[Tuple[int, int], str]] = decoded["ner"]["gold_ner"]
#             linked_ner: List[Dict[Tuple[int, int], float]] = decoded['linked']['decoded_spans']

#             assert ([len(x) for x in predicted_ner if len(x) != 0] == [len(x) for x in linked_ner if len(x) != 0]), breakpoint()
#             for x, y in zip(predicted_ner, linked_ner) :
#                 for k in x :
#                     assert type(y[k]) == float
#                     if type(x[k]) == str : x[k] = x[k].split('_')
#                     x[k] = "_".join(x[k] + [('True' if y[k] > linking_threshold else 'False')])

#             for x in gold_ner :
#                 for k in x :
#                     x[k] = "_".join(x[k])

            metadata = decoded["ner"]["metadata"]
            doc_ids: List[str] = [m["doc_key"] for m in metadata]
            assert len(set(doc_ids)) == 1
            para_ids: List[int] = [m["sentence_num"] for m in metadata]
            para_starts: List[int] = [int(m["start_pos_in_doc"]) for m in metadata]
            para_ends: List[int] = [int(m["end_pos_in_doc"]) for m in metadata]
            words: List[str] = [m["sentence"] for m in metadata]

#             relation_spans = decoded["relation"]["spans"]
#             relation_scores = decoded["relation"]["relation_scores"]
#             relation_doc_id = [m["doc_key"] for m in metadata]
#             assert len(set(relation_doc_id)) == 1
#             relation_doc_id = relation_doc_id[0]

#             if relation_doc_id not in documents_relations:
#                 documents_relations[relation_doc_id] = []

#             documents_relations[relation_doc_id].extend(
#                 [
#                     ((int(e1[0]), int(e1[1] + 1)), (int(e2[0]), int(e2[1] + 1)), float(round(relation_scores[i, j], 3)))
#                     for i, e1 in enumerate(relation_spans)
#                     for j, e2 in enumerate(relation_spans)
#                     if tuple(e1) != tuple(e2)
#                 ]
#             )

            # n_ary_relation_candidates = decoded['n_ary_relation']["candidates"]
            # n_ary_relation_gold = decoded['n_ary_relation']['gold']
            # n_ary_relation_scores = decoded['n_ary_relation']['scores']
            # map_coref_to_gold = metadata[0]['map_coref_keys']

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
#         for d in documents:
#             documents[d]["relation_scores"] = documents_relations[d]

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

        documents[k] = {"words": words, "paragraphs": paragraphs, "prediction": predictions, "gold": golds, "doc_id": k}

    return documents


def main():
    archive_folder = argv[1]
    test_file = argv[2]
    output_file = argv[3]
    cuda_device = int(argv[4])
    predict(archive_folder, test_file, output_file, cuda_device)


if __name__ == "__main__":
    main()
