echo "Predicting NER"
python scirex/predictors/predict_ner.py \
$scirex_archive \
$test_file \
$test_output_folder/ner_predictions.jsonl \
$cuda_device

echo "Predicting Salient Mentions"
python scirex/predictors/predict_salient_mentions.py \
$scirex_archive \
$test_output_folder/ner_predictions.jsonl \
$test_output_folder/salient_mentions_predictions.jsonl \
$cuda_device

echo "Predicting Coreference between mentions"
python scirex/predictors/predict_pairwise_coreference.py \
$scirex_coreference_archive \
$test_output_folder/ner_predictions.jsonl \
$test_output_folder/coreference_predictions.jsonl \
$cuda_device

echo "Predicting clusters"
python scirex/predictors/predict_clusters.py \
$test_output_folder/coreference_predictions.jsonl \
$test_output_folder/cluster_predictions.jsonl \
0.95

echo "Predicting Salient Clustering "
python scirex/predictors/predict_salient_clusters.py \
$test_output_folder/cluster_predictions.jsonl \
$test_output_folder/salient_mentions_predictions.jsonl \
$test_output_folder/salient_clusters_predictions.jsonl

echo "Predicting Relations End-to-End"
python scirex/predictors/predict_n_ary_relations.py \
$scirex_archive \
$test_output_folder/ner_predictions.jsonl \
$test_output_folder/salient_clusters_predictions.jsonl \
$test_output_folder/relations_predictions_with_graph_embeddings_e2e.jsonl \
$cuda_device


