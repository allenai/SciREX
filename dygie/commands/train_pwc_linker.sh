if [ $# -eq  0 ]
  then
    echo "No argument supplied for experiment name"
    exit 1
fi

export CONFIG_FILE=dygie/training_config/entity_linking.jsonnet

export CUDA_DEVICE=$CUDA_DEVICE

SEED=10034
PYTORCH_SEED=`expr $SEED / 10`
NUMPY_SEED=`expr $PYTORCH_SEED / 10`
export SEED=$SEED
export PYTORCH_SEED=$PYTORCH_SEED
export NUMPY_SEED=$NUMPY_SEED

export IS_LOWERCASE=true

export ERC_DATA_BASE_PATH=model_data/pwc_split_on_labeled
export TRAIN_PATH=$ERC_DATA_BASE_PATH/train.jsonl
export DEV_PATH=$ERC_DATA_BASE_PATH/dev.jsonl
export TEST_PATH=$ERC_DATA_BASE_PATH/test.jsonl

export OUTPUT_BASE_PATH=$OUTPUT_DIR/pwc_outputs/experiment_linker/$1/`date "+%Y%m%d-%H%M%S"`

allennlp train -s $OUTPUT_BASE_PATH --include-package dygie $CONFIG_FILE