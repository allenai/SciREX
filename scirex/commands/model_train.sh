DOCUMENT_FILTER=full \
USE_LSTM=true \
BERT_FINE_TUNE=pooler,11,10,9 \
bash scirex/commands/train_pwc_crf_n_ary.sh rnlw_${relation_cardinality}_${nw}_${lw}_${rw}_${cw}