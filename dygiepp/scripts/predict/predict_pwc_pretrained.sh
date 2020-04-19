# Make a prediction on the SciERC test set using the pretrained SciERC model.

# Make prediction directory if it doesn't exist.
if [ ! -d "./predictions" ]
then
    mkdir "./predictions"
fi

data_root="../model_data/pwc_split_on_sectioned/scierc_format_all/"

python ./dygie/commands/predict_dygie.py \
    ./models/pwc/model.tar.gz \
    ./$data_root/dev.jsonl \
    ./predictions/scierc_pwc_dev_all.jsonl \
    $CUDA_DEVICE
