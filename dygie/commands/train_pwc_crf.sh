if [ $# -eq  0 ]
  then
    echo "No argument supplied for experiment name"
    exit 1
fi

export BERT_VOCAB=$BERT_BASE_FOLDER/scivocab_uncased.vocab
export BERT_WEIGHTS=$BERT_BASE_FOLDER/scibert_scivocab_uncased.tar.gz

export CONFIG_FILE=dygie/training_config/pwc_config_crf.jsonnet

export CUDA_DEVICE=$CUDA_DEVICE

SEED=10034
PYTORCH_SEED=`expr $SEED / 10`
NUMPY_SEED=`expr $PYTORCH_SEED / 10`
export SEED=$SEED
export PYTORCH_SEED=$PYTORCH_SEED
export NUMPY_SEED=$NUMPY_SEED

export IS_LOWERCASE=true

export DATA_BASE_PATH=model_data/dataset_readers_paths

export TRAIN_DATASETS=pwc
export TRAIN_PATH=$DATA_BASE_PATH/train.json:$TRAIN_DATASETS
export DEV_PATH=$DATA_BASE_PATH/dev.json:pwc
export TEST_PATH=$DATA_BASE_PATH/test.json:pwc

export OUTPUT_BASE_PATH=${OUTPUT_DIR:-outputs/pwc_outputs/experiment_dygie_crf/$1}

allennlp train -s $OUTPUT_BASE_PATH --include-package dygie $CONFIG_FILE