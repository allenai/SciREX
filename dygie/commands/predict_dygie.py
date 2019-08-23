import os
import json
from sys import argv

import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
logging.info("Loading Allennlp Modules")

from dygie.commands.predict_spans import predict as predict_spans
from dygie.commands.predict_coref import predict as predict_coref

logging.info("Loaded Allennlp Modules")


def predict(span_archive_folder, coref_archive_folder, test_file, output_folder, cuda_device) :
    os.makedirs(output_folder, exist_ok=True)
    span_file = os.path.join(output_folder, 'spans.jsonl')
    coref_file = os.path.join(output_folder, 'coref.jsonl')

    coref_threshold = json.load(open(coref_archive_folder + '/metrics.json'))['best_validation_threshold']
    relation_threshold = json.load(open(span_archive_folder + '/metrics.json'))['best_validation_rel_threshold']

    logging.info("Predicting ")
    
    predict_spans(span_archive_folder, test_file, span_file, cuda_device)
    predict_coref(coref_archive_folder, span_file, coref_file, cuda_device)

    coref_output = [json.loads(line) for line in open(coref_file)]
    ner_output = [json.loads(line) for line in open(span_file)]
    ner_output = {x['doc_id']: x for x in ner_output}
    for d in coref_output :
        ner_output[d['doc_id']]['coref_prediction'] = d['coref_prediction']
        ner_output[d['doc_id']]['coref_gold'] = d['coref_gold']

    for k, d in ner_output.items() :
        d['coref_threshold'] = coref_threshold
        d['relation_threshold'] = relation_threshold

    with open(os.path.join(output_folder, 'combined.jsonl'), 'w') as f :
        f.write('\n'.join([json.dumps(d) for d in ner_output.values()]))

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