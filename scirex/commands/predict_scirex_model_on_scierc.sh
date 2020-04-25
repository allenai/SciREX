export test_file=dygiepp/data/scierc/processed_data/scirex/test.jsonl
mkdir -p outputs/scirex_on_scierc_results
export output_folder=outputs/scirex_on_scierc_results

# python scirex/predictors/predict_ner.py $scirex_archive \
# $test_file \
# $output_folder/test_ner_predictions.jsonl \
# $cuda_device

# python scirex/predictors/predict_pairwise_coreference.py \
# $scirex_coreference_archive \
# $output_folder/test_ner_predictions.jsonl \
# $output_folder/test_coreference_predictions.jsonl \
# $cuda_device

python scirex/predictors/predict_clusters.py \
$output_folder/test_coreference_predictions.jsonl \
$output_folder/test_cluster_predictions.jsonl \
0.93