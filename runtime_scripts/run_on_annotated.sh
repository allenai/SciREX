for i in `eval echo {${START}..${END}..20}`;
do
    python dygie/commands/predict_dygie.py \
    outputs/pwc_outputs/experiment_dygie_crf/only_ner_for_annotation/ \
    outputs/pwc_outputs/experiment_linker/berty-bert/ \
    model_data/dataset_readers_paths/${i}_unannotated.json:pwc \
    outputs/unannotated_results_folder/${i}_unannotated/ ${CUDA_DEVICE};
done;
