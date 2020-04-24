import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


def generate_matrix_for_document(document, span_field, matrix_field) :
    span2idx = {tuple(k):i for i, k in enumerate(document[span_field])}
    matrix = np.zeros((len(span2idx), len(span2idx)))
    for e1, e2, score in document[matrix_field] :
        matrix[span2idx[tuple(e1)], span2idx[tuple(e2)]] = score
        
    return matrix


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

    return clusters
