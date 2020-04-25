import pandas as pd
from sys import argv

from scirex.predictors.utils import span_match
from scirex_utilities.json_utilities import load_jsonl


def overlap_score(cluster_1, cluster_2):
    matched = 0
    for s1 in cluster_1:
        matched += 1 if any([span_match(s1, s2) > 0.5 for s2 in cluster_2]) else 0

    return matched / len(cluster_1)


def compute_metrics(predicted_clusters, gold_clusters):
    matched_predicted = []
    matched_gold = []
    for i, p in enumerate(predicted_clusters):
        for j, g in enumerate(gold_clusters):
            if overlap_score(p, g) > 0.5:
                matched_predicted.append(i)
                matched_gold.append(j)

    matched_predicted = set(matched_predicted)
    matched_gold = set(matched_gold)

    metrics = {
        "p": len(matched_predicted) / (len(predicted_clusters) + 1e-7),
        "r": len(matched_gold) / (len(gold_clusters) + 1e-7),
    }
    metrics["f1"] = 2 * metrics["p"] * metrics["r"] / (metrics["p"] + metrics["r"] + 1e-7)

    return metrics


def score_scirex_model(predictions, gold_data):
    gold_data = {x["doc_id"]: list(x["coref"].values()) for x in gold_data}
    predictions = {x["doc_id"]: list(x["clusters"].values()) for x in predictions}

    all_metrics = []
    for p, pc in predictions.items():
        metrics = compute_metrics(pc, gold_data[p])
        all_metrics.append(metrics)

    all_metrics = pd.DataFrame(all_metrics)
    print(all_metrics.describe())


def score_dygie_model(predictions, gold_data):
    gold_data = {x["doc_id"]: list(x["coref"].values()) for x in gold_data}
    predictions = {x["doc_key"]: [[(s, e + 1) for s, e in c] for c in x["predicted_clusters"]] for x in predictions}

    all_metrics = []
    for p, pc in predictions.items():
        metrics = compute_metrics(pc, gold_data[p])
        all_metrics.append(metrics)

    all_metrics = pd.DataFrame(all_metrics)
    print(all_metrics.describe())


if __name__ == "__main__":
    print("DyGIE")
    score_dygie_model(load_jsonl(argv[2]), load_jsonl(argv[1]))

    print("SciREX")
    score_scirex_model(load_jsonl(argv[3]), load_jsonl(argv[1]))
