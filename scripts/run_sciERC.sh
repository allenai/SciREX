export NER_TRAIN_DATA_PATH=data/sciERC_processed_data/conll/train.txt
export NER_DEV_DATA_PATH=data/sciERC_processed_data/conll/dev.txt
export NER_TEST_DATA_PATH=data/sciERC_processed_data/conll/test.txt

allennlp train -s outputs/sciERC_outputs/experiment_1 scripts/sciERC_model_configs/ner.jsonnet
