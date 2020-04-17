import json
import os

from scirex.models.global_analysis.clustering import do_clustering


def predict(results_folder, output_file, coreference_threshold):
    coref_file = os.path.join(results_folder, "coref.jsonl")
    span_file = os.path.join(results_folder, "spans.jsonl")
    coref_documents = [json.loads(line) for line in open(coref_file)]
    documents = [json.loads(line) for line in open(span_file)]
    documents = {x['doc_id']:x for x in documents}
    for d in coref_documents :
        documents[d['doc_id']].update(d)

    documents = list(documents.values())

    for v in documents:
        v["prediction"] = {tuple(x["span"]): x["label"] for x in v["prediction"]}
        v["gold"] = {tuple(x["span"]): x["label"] for x in v["gold"]}

    new_documents = []
    new_gold_documents = []
    for v in documents:
        if len(v['coref_prediction']) != 0 :
            clusters = do_clustering(v, "prediction", "coref_prediction", plot=True, threshold=coreference_threshold)
            coref_clusters = {str(i): v["spans"] for i, v in enumerate(clusters)}
        else :
            coref_clusters = {}

        gold_clusters = do_clustering(v, "gold", "coref_gold", plot=True)
        gold_coref_clusters = {str(i): v["spans"] for i, v in enumerate(gold_clusters)}

        doc = {
            "doc_id": v["doc_id"],
            "words": v["words"],
            "sections": v["sections"],
            "sentences" : v['sentences'],
            "ner": [(k[0], k[1], x) for k, x in v["prediction"].items()],
            "n_ary_relations": [],
            "coref": coref_clusters,
        }

        new_documents.append(doc)

        new_gold_documents.append({
            "doc_id": v["doc_id"],
            "words": v["words"],
            "sections": v["sections"],
            "sentences" : v['sentences'],
            "ner": [(k[0], k[1], x) for k, x in v["gold"].items()],
            "n_ary_relations": [],
            "coref": gold_coref_clusters,
        })

    with open(output_file + ".predicted_mentions", "w") as f:
        f.write("\n".join([json.dumps(line) for line in new_documents]))

    with open(output_file + ".gold_mentions", "w") as f:
        f.write("\n".join([json.dumps(line) for line in new_gold_documents]))
