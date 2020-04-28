# Make a prediction on the SciERC test set using the pretrained SciERC model.

# Make prediction directory if it doesn't exist.
if [ ! -d "./predictions" ]
then
    mkdir "./predictions"
fi

data_root="../scirex_dataset/release_data_in_scierc_format/"

python ./dygie/commands/predict_dygie.py \
    $model \
    ./$data_root/test.jsonl \
    ./predictions/$2 \
    $1
