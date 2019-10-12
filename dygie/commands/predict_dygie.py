import os
import json
from sys import argv

import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
logging.info("Loading Allennlp Modules")

from dygie.commands.predict_spans import predict as predict_spans
from dygie.commands.predict_ner import predict as predict_ner
from dygie.commands.predict_coref import predict as predict_coref
from dygie.commands.predict_clusters import predict as predict_clusters
from dygie.commands.predict_n_ary_relations import predict as predict_n_ary_relations

logging.info("Loaded Allennlp Modules")

def predict(span_archive_folder, coref_archive_folder, test_file, output_folder, cuda_device) :
    os.makedirs(output_folder, exist_ok=True)
    span_file = os.path.join(output_folder, 'spans.jsonl')
    coref_file = os.path.join(output_folder, 'coref.jsonl')
    cluster_file = os.path.join(output_folder, 'clusters.jsonl')
    n_ary_relations_file = os.path.join(output_folder, 'n_ary_relations.jsonl')

    '''
    coref_threshold = json.load(open(coref_archive_folder + '/metrics.json'))['best_validation_threshold']
    relation_threshold = json.load(open(span_archive_folder + '/metrics.json'))['best_validation__rel_threshold']
    '''

    logging.info("Predicting ")
    
    # predict_ner(span_archive_folder, test_file, span_file, cuda_device)
    # predict_coref(coref_archive_folder, span_file, coref_file, cuda_device)
    # predict_clusters(output_folder, cluster_file)
    predict_n_ary_relations(span_archive_folder, cluster_file, n_ary_relations_file, cuda_device)
    

def main():
    span_archive_folder = argv[1]
    coref_archive_folder = argv[2]
    test_file = argv[3]
    output_folder = argv[4]
    cuda_device = int(argv[5])
    predict(span_archive_folder, coref_archive_folder, test_file, output_folder, cuda_device)


if __name__ == "__main__":
    logging.info("Starting Script")
    main()
