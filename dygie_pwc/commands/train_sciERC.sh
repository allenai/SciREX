export BERT_VOCAB=/net/nfs.corp/s2-research/scibert/scivocab_uncased.vocab
export BERT_WEIGHTS=/net/nfs.corp/s2-research/scibert/scibert_scivocab_uncased.tar.gz

export CONFIG_FILE=dygie/training_config/sciERC_config.jsonnet

export CUDA_DEVICE=0

SEED=13270
PYTORCH_SEED=`expr $SEED / 10`
NUMPY_SEED=`expr $PYTORCH_SEED / 10`
export SEED=$SEED
export PYTORCH_SEED=$PYTORCH_SEED
export NUMPY_SEED=$NUMPY_SEED

export IS_LOWERCASE=true

export ERC_DATA_BASE_PATH=data/sciERC_processed_data/json
export TRAIN_PATH=$ERC_DATA_BASE_PATH/train.json
export DEV_PATH=$ERC_DATA_BASE_PATH/dev.json
export TEST_PATH=$ERC_DATA_BASE_PATH/test.json

export OUTPUT_BASE_PATH=outputs/sciERC_outputs/experiment_BERT_DyGIE_bert_final_layer_and_lstm/`date "+%Y%m%d-%H%M%S"`

allennlp train -s $OUTPUT_BASE_PATH --include-package dygie $CONFIG_FILE