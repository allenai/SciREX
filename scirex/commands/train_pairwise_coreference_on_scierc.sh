if [ $# -eq  0 ]
  then
    echo "No argument supplied for experiment name"
    exit 1
fi

export BERT_VOCAB=$BERT_BASE_FOLDER/vocab.txt
export BERT_WEIGHTS=$BERT_BASE_FOLDER/weights.tar.gz

export CONFIG_FILE=scirex/training_config/pairwise_coreference.jsonnet

export CUDA_DEVICE=$CUDA_DEVICE

export IS_LOWERCASE=true

export ERC_DATA_BASE_PATH=dygiepp/data/scierc/processed_data/scirex/coref_version
export TRAIN_PATH=$ERC_DATA_BASE_PATH/train.jsonl
export DEV_PATH=$ERC_DATA_BASE_PATH/dev.jsonl
export TEST_PATH=$ERC_DATA_BASE_PATH/test.jsonl

export OUTPUT_BASE_PATH=${OUTPUT_DIR:-outputs/pwc_outputs/experiment_coreference_on_scierc/$1/}

python -m allennlp.run train -s $OUTPUT_BASE_PATH --include-package scirex --force $CONFIG_FILE
