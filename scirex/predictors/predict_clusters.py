import json
from scirex.models.global_analysis.clustering import do_clustering


def predict(coreference_scores_file, output_file, coreference_threshold):
    '''
    coreference_scores_file (jsonl) -
    {
        'doc_id' : str,
        'pairwise_coreference_scores' : List[(s_1, e_1), (s_2, e_2), float (3 sig. digits) in [0, 1]]
    }

    output_file (jsonl) - 
    {
        'doc_id' : str,
        'spans' : List[Tuple[int, int]]
        'clusters' : Dict[str, List[Tuple[int, int]]]
    }
    '''
    documents = [json.loads(line) for line in open(coreference_scores_file)]
    for doc in documents:
        doc["spans"] = sorted(
            list(
                set(
                    [tuple(x[0]) for x in doc["pairwise_coreference_scores"]]
                    + [tuple(x[1]) for x in doc["pairwise_coreference_scores"]]
                )
            )
        )

    cluster_outputs = []
    for doc in documents:
        clusters = do_clustering(
            doc, "spans", "pairwise_coreference_scores", plot=True, threshold=coreference_threshold
        )
        coref_clusters = {str(i): v["spans"] for i, v in enumerate(clusters)}

        cluster_outputs.append({'doc_id' : doc['doc_id'], 'spans' : doc['spans'], 'clusters' : coref_clusters})

    with open(output_file, "w") as f:
        f.write("\n".join([json.dumps(line) for line in cluster_outputs]))
