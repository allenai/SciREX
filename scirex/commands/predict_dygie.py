import os
import shutil
import logging
import json

logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s", level=logging.INFO)
logging.info("Loading Allennlp Modules")

from scirex.commands.predict_ner import predict as predict_ner
from scirex.commands.predict_coref import predict as predict_coref
from scirex.commands.predict_clusters import predict as predict_clusters
from scirex.commands.predict_n_ary_relations import predict as predict_n_ary_relations
from scirex.commands.predict_links import predict as predict_links

logging.info("Loaded Allennlp Modules")


def predict(
    span_archive_folder,
    coref_archive_folder,
    link_archive_folder,
    relation_archive_folder,
    test_file,
    output_folder,
    cuda_device,
):
    os.makedirs(output_folder, exist_ok=True)
    span_file = os.path.join(output_folder, "spans.jsonl")
    coref_file = os.path.join(output_folder, "coref.jsonl")
    cluster_file = os.path.join(output_folder, "clusters.jsonl")
    links_file = os.path.join(output_folder, "links.jsonl")
    n_ary_relations_file = os.path.join(output_folder, "n_ary_relations.jsonl")

    logging.info("Predicting ")

    coreference_threshold = json.load(open(os.path.join(coref_archive_folder, 'metrics.json')))['best_validation_threshold']

    # predict_ner(span_archive_folder, test_file, span_file, cuda_device)
    # predict_coref(coref_archive_folder, span_file, coref_file, cuda_device)
    # predict_clusters(output_folder, cluster_file, coreference_threshold)
    # predict_links(link_archive_folder, cluster_file + '.gold', links_file + '.gold', cuda_device)
    # predict_n_ary_relations(relation_archive_folder, cluster_file + '.gold.linked', n_ary_relations_file + '.gold', cuda_device)

    # shutil.copyfile(test_file, cluster_file + '.orig')
    # predict_links(link_archive_folder, cluster_file, links_file, cuda_device)
    # predict_n_ary_relations(relation_archive_folder, cluster_file + '.linked', n_ary_relations_file, cuda_device)

    predict_n_ary_relations(relation_archive_folder, cluster_file + '.map_from_gold', n_ary_relations_file + '.map_from_gold', cuda_device)


def main(args):
    span_archive_folder = args.ner
    coref_archive_folder = args.coref
    relation_archive_folder = args.relation
    link_archive_folder = args.link
    test_file = args.file
    output_folder = args.output_dir
    cuda_device = int(args.cuda_device)
    predict(
        span_archive_folder,
        coref_archive_folder,
        link_archive_folder,
        relation_archive_folder,
        test_file,
        output_folder,
        cuda_device,
    )


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ner")
parser.add_argument("--coref")
parser.add_argument("--relation")
parser.add_argument("--link")
parser.add_argument("--file")
parser.add_argument("--output-dir")
parser.add_argument("--cuda-device")


if __name__ == "__main__":
    logging.info("Starting Script")
    args = parser.parse_args()
    main(args)
