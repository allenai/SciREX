# SciREX : A Challenge Dataset for Document-Level Information Extraction

Our data can be found here : https://github.com/allenai/SciREX/blob/master/scirex_dataset/release_data.tar.gz
You can also browse the dataset at - https://allenai.github.io/SciREX/

It contains 3 files - {train, dev, test}.jsonl

Each file contains one document per line in format  - 

```python
{
    "doc_id" : str,
    "words" : List[str],
    "sentences" : List[Span],
    "sections" : List[Span],
    "ner" : List[TypedMention],
    "coref" : Dict[EntityName, List[Span]],
    "n_ary_relations" : Dict[EntityType, EntityName],
    "method_subrelations" : Dict[EntityName, List[Tuple[Span, SubEntityName]]]
}

Span = Tuple[int, int] # Inclusive start and Exclusive end index
TypedMention = Tuple[int, int, EntityType]
EntityType = Union["Method", "Metric", "Task", "Material"]
EntityName = str
```

<hr>

Installation
============

1. `conda create -n scirex python=3.7`
2. `pip install -r requirements.txt`
3. `python -m spacy download en`

Training SciREX baseline Model
=================

1. Extract the dataset files in folder `tar -xvzf scirex_dataset/release_data.tar.gz scirex_data/release_data`
2. Run `CUDA_DEVICE=<cuda-device-num> bash scirex/commands/train_scirex_model.sh main` to train main scirex model
3. Run `CUDA_DEVICE=<cuda-device-num> bash scirex/commands/train_pairwise_coreference.sh main` to train secondary coreference model.

Generating Predictions
======================


```bash
scirex_archive=outputs/pwc_outputs/experiment_scirex_full/main \
scirex_coreference_archive=outputs/pwc_outputs/experiment_pairwise_coreference/main \
cude_device=<cuda-device-num> \
bash scirex/commands/predict_scirex_model.sh
```
