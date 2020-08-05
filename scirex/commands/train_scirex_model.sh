if [ $# -eq  0 ]
  then
    echo "No argument supplied for experiment name"
    exit 1
fi

export BERT_VOCAB=$BERT_BASE_FOLDER/vocab.txt
export BERT_WEIGHTS=$BERT_BASE_FOLDER/weights.tar.gz

export CONFIG_FILE=scirex/training_config/scirex_full.jsonnet

export CUDA_DEVICE=$CUDA_DEVICE

export IS_LOWERCASE=true

export DATA_BASE_PATH=scirex_dataset/release_data

export TRAIN_PATH=$DATA_BASE_PATH/train.jsonl
export DEV_PATH=$DATA_BASE_PATH/dev.jsonl
export TEST_PATH=$DATA_BASE_PATH/test.jsonl

export OUTPUT_BASE_PATH=${OUTPUT_DIR:-outputs/pwc_outputs/experiment_scirex_full/$1}

export bert_fine_tune=10,11,pooler

nw=1 lw=1 rw=1 em=false \
relation_cardinality=4 \
allennlp train -s $OUTPUT_BASE_PATH --include-package scirex $RECOVER $CONFIG_FILE
