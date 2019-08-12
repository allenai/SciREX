from typing import List, Tuple
from allennlp.data.vocabulary import Vocabulary
import json
import torch

def _doc_bioul_to_spans(doc:List[str], vocab:Vocabulary) -> List[Tuple[int, int, int]]:
    '''Given bioul predictions of one document, return entities in the span format'''
    spans = []
    for i, l in enumerate(doc):
        if l != 'O':
            span_label = l[2:]
            span_label_index = vocab.get_token_index(span_label, namespace='span_labels')  # TODO: is this the right namespace?
        if l.startswith('U'):
            spans.append((i, i+1, span_label_index))
        elif l.startswith('B'):
            start_index = i
        elif l.startswith('L'):
            spans.append((start_index, i+1, span_label_index))
    return spans

def batched_bioul_to_span_tesnors(docs:List[List[str]], vocab:Vocabulary) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''Given bioul predictions of a list of documents, return three tensors, spans, span labels, and mask'''
    doc_spans = []
    for doc in docs:
        spans = _doc_bioul_to_spans(doc, vocab)
        doc_spans.append(spans)
    max_spans_count = max([len(s) for  s in doc_spans])

    # padding
    for spans in doc_spans:
        for _ in range(max_spans_count - len(spans)):
            spans.append((0, 0, 0))

    spans_lists = [[span[:2] for span in spans] for spans in doc_spans]
    span_labels_lists = [[span[2] for span in spans] for spans in doc_spans]
    span_masks_lists = [[0 if span[2] == 0 else 1 for span in spans] for spans in doc_spans]

    return torch.tensor(spans_lists), torch.tensor(span_labels_lists), torch.tensor(span_masks_lists)

def main():
    infilename = 'test/fixtures/bioul_to_span.json'
    with open(infilename) as f:
        d = json.load(f)

    docs = d['tag']
    vocab = Vocabulary()
    vocab.add_token_to_namespace('O', namespace='span_labels')  # reserved label for no-entity
    for doc in docs:
        for label in doc:
            if label != 'O':
                span_label = label[2:]  # drop the first two character because they are not useful for span labels
                vocab.add_token_to_namespace(span_label, namespace='span_labels')  # TODO: is this the right namespace?

    # this function is expecting the vocab is already initialized with span labels
    batched_bioul_to_span_tesnors(docs, vocab)

if __name__ == "__main__":
    main()