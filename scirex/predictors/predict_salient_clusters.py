import json


def predict(clusters_file, saliency_file, output_file):
    clusters = [json.loads(line) for line in open(clusters_file)]
    saliency = {item["doc_id"]: item for item in [json.loads(line) for line in open(saliency_file)]}

    with open(output_file, "w") as f:
        for doc in clusters:
            sdoc = saliency[doc["doc_id"]]

            spans = set(list(map(tuple, doc["spans"])))
            assert all([tuple(span[0], span[1]) in spans for span in sdoc["saliency"]])

            salient_spans = set([tuple(span[0], span[1]) for span in sdoc["saliency"] if span[2] == 1])

            salient_clusters = {}
            for cluster, cluster_spans in doc["clusters"].items():
                cluster_spans = list(map(tuple, cluster_spans))
                if len(set(cluster_spans) & salient_spans) > 0:
                    salient_clusters[cluster] = cluster_spans

            f.write(json.dumps({"doc_id": doc["doc_id"], "clusters": salient_clusters, "spans" : doc['spans']}) + "\n")
