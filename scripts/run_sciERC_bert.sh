export BERT_VOCAB=/net/nfs.corp/s2-research/scibert/scivocab_uncased.vocab
export BERT_WEIGHTS=/net/nfs.corp/s2-research/scibert/scibert_scivocab_uncased.tar.gz

export CONFIG_FILE=scripts/sciERC_model_configs/ner_bert_nofinetune.jsonnet

export CUDA_DEVICE=0,1

SEED=13270
PYTORCH_SEED=`expr $SEED / 10`
NUMPY_SEED=`expr $PYTORCH_SEED / 10`
export SEED=$SEED
export PYTORCH_SEED=$PYTORCH_SEED
export NUMPY_SEED=$NUMPY_SEED

export IS_LOWERCASE=true

export ERC_DATA_BASE_PATH=data/sciERC_processed_data/conll
export TRAIN_PATH=$ERC_DATA_BASE_PATH/train.txt
export DEV_PATH=$ERC_DATA_BASE_PATH/dev.txt
export TEST_PATH=$ERC_DATA_BASE_PATH/test.txt

export OUTPUT_BASE_PATH=outputs/sciERC_outputs/experiment_BERT_nofinetune/`date "+%Y%m%d-%H%M%S"`

allennlp train -s $OUTPUT_BASE_PATH $CONFIG_FILE
