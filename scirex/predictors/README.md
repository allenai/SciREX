Since we have multiple tasks, we do prediction step by step .

1. Main-file -> predict_ner.py -> ner
2. ner -> predict_saliency.py -> salient_mentions
3. ner -> predict_pairwise_coreference.py -> pc scores
4. pc scores -> predict_clusters.py -> clusters
5. clusters, salient_mentions -> predict_salient_clusters.py -> salient_clusters
6. salient_clusters, ner -> relations