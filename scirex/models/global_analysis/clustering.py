import json
import numpy as np
import matplotlib.pyplot as plt
from scripts.entity_utils import *

from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

from scirex.models.global_analysis import *

def cluster_with_clustering(matrix, threshold, plot=True) :
    scores = []
    matrix = (matrix + matrix.T) + np.eye(*matrix.shape)
    for n in range(2, matrix.shape[0] if matrix.shape[0] > 2 else 3) :
        clustering = AgglomerativeClustering(n_clusters=n, linkage='complete', affinity='precomputed').fit(1 - matrix)
        if matrix.shape[0] > 2 :
            scores.append(silhouette_score(1 - matrix, clustering.labels_, metric='precomputed'))
        else :
            scores.append(1)
    try :
        if False :
            plt.plot(range(2, matrix.shape[0]), scores)
        best_score = max(scores)
    except :
        breakpoint()
    best_n = scores.index(best_score) + 2
    clustering = AgglomerativeClustering(n_clusters=best_n, linkage='complete', affinity='precomputed').fit(1 - matrix)
    return clustering.n_clusters_, clustering.labels_

def cluster_with_dbscan_clustering(matrix, threshold, plot=True) :
    scores = []
    matrix = (matrix + matrix.T) + np.eye(*matrix.shape)
    eps_space = list(np.linspace(0.0001, 1, 100))
    for eps in eps_space :
        clustering = DBSCAN(eps=eps, metric='precomputed').fit(1 - matrix)
        try :
            scores.append(silhouette_score(1 - matrix, clustering.labels_, metric='precomputed'))
        except :
            scores.append(0.0)
    if plot :
        plt.plot(eps_space, scores)
    best_score = max(scores)
    best_eps = eps_space[scores.index(best_score)]
    clustering = DBSCAN(eps=best_eps, metric='precomputed').fit(1 - matrix)
    labels = clustering.labels_
    max_label = max(clustering.labels_) + 1
    for i in range(len(labels)) :
        if labels[i] == -1 :
            labels[i] = max_label
            max_label += 1

    return max(labels) + 1, labels

from scipy.sparse.csgraph import connected_components
def cluster_with_connected_components(matrix, threshold, plot) :
    graph = ((matrix + matrix.T) > threshold).astype(int)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    return n_components, labels

def map_back_to_spans(document, span_field, labels) :
    idx2span = {i:tuple(k) for i, k in enumerate(document[span_field])}
    span_to_label_map = {}
    for i, span in idx2span.items() :
        span_to_label_map[span] = labels[i]
    return span_to_label_map

def do_clustering(document, span_field, coref_field, plot=True, threshold=0.5) :
    matrix = generate_matrix_for_document(document, span_field, coref_field)
    n_clusters, cluster_labels = cluster_with_clustering(matrix, threshold, plot)
    span_to_cluster_label = map_back_to_spans(document, span_field, cluster_labels)

    clusters = [{'spans' : [], 'words': set(), 'types' : set()} for _ in range(n_clusters)]
    for s, l in span_to_cluster_label.items() :
        clusters[l]['spans'].append(s)
        clusters[l]['words'].add(" ".join(document['words'][s[0]:s[1]]))
        clusters[l]['types'].add(document[span_field][s])
        
    for c in clusters :
        strings = [s for s in list(c['words'])]
        lengths = [len(s) for s in strings]
        c['type'] = list(c['types'])[0]

    return clusters

def get_linked_clusters(clusters) :
    linked_clusters = []
    for i, c in enumerate(clusters) :
        c_types = [x.split('_')[2] for x in c['types']]
        if 'True' in c_types :
            linked_clusters.append(i)
    return linked_clusters

# from allennlp.training.metrics.conll_coref_scores import ConllCorefScores
# def evaluate_clustering_for_single_document(document, scorer) :
#     clusters, spl, cluster_labels, linked_clusters = do_clustering(document, 'prediction', 'coref_prediction', plot=False)
#     true_clusters = [c for c in document['true_coref'].values() if len(c) > 0]
#     gold_clusters, mention_to_gold = scorer.get_gold_clusters(true_clusters)
#     predicted_clusters, mention_to_predicted = scorer.get_gold_clusters([x['spans'] for i, x in enumerate(clusters) if i in linked_clusters])
#     for s in scorer.scorers :
#         s.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)

# def evaluate_for_all_document(documents) :
#     scorer = ConllCorefScores()
#     for d in documents :
#         evaluate_clustering_for_single_document(d, scorer)

#     return scorer.get_metric(reset=True)

from scirex.training.evaluation import match_clusters
def match_cluster_to_true(cluster, entities, threshold):
    if len(cluster['words']) == 0 :
        return None
    scores = match_clusters(list(cluster['words']), entities).max(0)
    best_score, best_entity = scores.max(), scores.argmax()

    if best_score >= threshold :
        return entities[best_entity].replace(' ', '_')
    else:
        return None

def map_all_clusters_to_true(document, cluster_matching_thresholds) :
    true_e = [x.replace('_', ' ') for x in list(document['gold_clusters'].keys())]
    pred_e = document['clusters']
    for k, c in document['gold_clusters'].items() :
        c['matched'] = match_cluster_to_true(c, true_e, cluster_matching_thresholds[document['gold_to_type'][k]])

    for c in pred_e :
        c['matched'] = match_cluster_to_true(c, true_e, cluster_matching_thresholds[c['type']])

def map_gold_clusters_to_true(document, cluster_matching_thresholds) :
    for k, c in document['gold_clusters'].items() :
        c['matched'] = match_cluster_to_true(c, [k.replace('_', ' ')], cluster_matching_thresholds[document['gold_to_type'][k]])