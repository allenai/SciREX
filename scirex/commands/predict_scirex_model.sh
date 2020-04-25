export test_file=scirex_dataset/release_data/test.jsonl

python scirex/predictors/predict_ner.py $scirex_archive $test_file test_ner_predictions.jsonl $cuda_device

python scirex/predictors/predict_salient_mentions.py $scirex_archive test_ner_predictions.jsonl test_salient_mentions_predictions.jsonl $cuda_device

python scirex/predictors/predict_pairwise_coreference.py $scirex_coreference_archive test_ner_predictions.jsonl test_coreference_predictions.jsonl $cuda_device

python scirex/predictors/predict_clusters.py test_coreference_predictions.jsonl test_cluster_predictions.jsonl 0.95

python scirex/predictors/predict_salient_clusters.py test_cluster_predictions.jsonl test_salient_mentions_predictions.jsonl test_salient_clusters_predictions.jsonl

python scirex/predictors/predict_salient_clusters_using_gold.py test_cluster_predictions.jsonl $test_file test_salient_clusters_predictions_using_gold.jsonl

python scirex/predictors/predict_n_ary_relations.py \
outputs/pwc_outputs/experiment_scirex_full/everything/ \
test_ner_predictions.jsonl \
test_salient_clusters_predictions.jsonl \
test_relations_predictions.jsonl \
$cuda_device

python scirex/predictors/predict_n_ary_relations.py \
outputs/pwc_outputs/experiment_scirex_full/everything/ \
test_ner_predictions.jsonl \
test_salient_clusters_predictions_using_gold.jsonl \
test_relations_predictions_gold_salient_clusters.jsonl \
$cuda_device

python scirex/evaluation_scripts/scirex_relation_evaluate.py \
--gold-file $test_file \
--ner-file test_ner_predictions.jsonl \
--clusters-file test_salient_clusters_predictions.jsonl \
--relations-file test_relations_predictions.jsonl

python scirex/evaluation_scripts/scirex_relation_evaluate.py \
--gold-file $test_file \
--ner-file test_ner_predictions.jsonl \
--clusters-file test_salient_clusters_predictions_using_gold.jsonl \
--relations-file test_relations_predictions_gold_salient_clusters.jsonl