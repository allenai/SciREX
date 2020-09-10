export test_file=scirex_dataset/release_data/test.jsonl
export test_output_folder=test_outputs/

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

echo "Predicitng Salient Clusters using gold clusters as filter"
python scirex/predictors/predict_salient_clusters_using_gold.py \
$test_output_folder/cluster_predictions.jsonl \
$test_file \
$test_output_folder/salient_clusters_predictions_using_gold.jsonl

echo "Predicting Relations End-to-End"
python scirex/predictors/predict_n_ary_relations.py \
$scirex_archive \
$test_output_folder/ner_predictions.jsonl \
$test_output_folder/salient_clusters_predictions.jsonl \
$test_output_folder/relations_predictions.jsonl \
$cuda_device

echo "Predicting relations End-to-End with gold cluster filtering"
python scirex/predictors/predict_n_ary_relations.py \
$scirex_archive \
$test_output_folder/ner_predictions.jsonl \
$test_output_folder/salient_clusters_predictions_using_gold.jsonl \
$test_output_folder/relations_predictions_gold_salient_clusters.jsonl \
$cuda_device

echo "Predicting Relations on gold clusters"
python scirex/predictors/predict_n_ary_relations.py \
$scirex_archive \
$test_file \
$test_file \
$test_output_folder/relations_predictions_gold_clusters.jsonl \
$cuda_device

echo "Evaluating on all Predicted steps "
python scirex/evaluation_scripts/scirex_relation_evaluate.py \
--gold-file $test_file \
--ner-file $test_output_folder/ner_predictions.jsonl \
--clusters-file $test_output_folder/salient_clusters_predictions.jsonl \
--relations-file $test_output_folder/relations_predictions.jsonl


echo "Evaluating on all predicted steps with filtering using gold salient clusters"
python scirex/evaluation_scripts/scirex_relation_evaluate.py \
--gold-file $test_file \
--ner-file $test_output_folder/ner_predictions.jsonl \
--clusters-file $test_output_folder/salient_clusters_predictions_using_gold.jsonl \
--relations-file $test_output_folder/relations_predictions_gold_salient_clusters.jsonl
