# SciREX : A Challenge Dataset for Document-Level Information Extraction

Our data can be found here : https://github.com/allenai/SciREX/blob/master/scirex_dataset/release_data.tar.gz

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
    "n_ary_subrelations" : Dict[EntityType, EntityName],
    "method_subrelations" : Dict[EntityName, List[Tuple[Span, SubEntityName]]]
}

Span = Tuple[int, int]
TypedMention = Tuple[int, int, EntityType]
EntityType = Union["Method", "Metric", "Task", "Material"]
EntityName = str
```

<hr>



