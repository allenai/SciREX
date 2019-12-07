function train {
    DOCUMENT_FILTER=full \
    USE_LSTM=true \
    BERT_FINE_TUNE=pooler,11,10 \
    bash dygie/commands/train_pwc_crf_n_ary.sh rnlw_${relation_cardinality}_${nw}_${lw}_${rw}
}

relation_cardinality=3 nw=1 lw=1 rw=1 train
relation_cardinality=2 nw=1 lw=1 rw=1 train
relation_cardinality=3 nw=0 lw=0 rw=1 train
relation_cardinality=2 nw=0 lw=0 rw=1 train