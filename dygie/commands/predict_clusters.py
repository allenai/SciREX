from sys import argv
import json
import os

from dygie.models.global_analysis.clustering import do_clustering

def predict(results_folder, output_file) :
    coref_file = os.path.join(results_folder, "coref.jsonl")
    documents = [json.loads(line) for line in open(coref_file)]

    for v in documents :
        v['prediction'] = {tuple(x['span']):x['label'] for x in v['prediction']}
        v['gold'] = {tuple(x['span']):x['label'] for x in v['gold']}

    new_documents = []
    for v in documents :
        clusters, span_to_cluster_label, cluster_labels = do_clustering(v, 'prediction', 'coref_prediction', plot=True)
        coref_clusters = {
            str(i):v['spans'] for i, v in enumerate(clusters)
        }

        doc = {
            'doc_id' : v['doc_id'],
            'words' : v['words'],
            'sections' : v['paragraphs'],
            'ner' : [(k[0], k[1], x) for k, x in v['prediction'].items()],
            'n_ary_relations' : [],
            'coref' : coref_clusters
        }

        new_documents.append(doc)

    with open(output_file, 'w') as f :
        f.write('\n'.join([json.dumps(line) for line in new_documents]))

def main():
    result_folder = argv[1]
    output_file = argv[3]
    predict(result_folder, output_file)


if __name__ == "__main__":
    main()