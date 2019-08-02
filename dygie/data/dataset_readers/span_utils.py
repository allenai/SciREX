from allennlp.data.dataset_readers.dataset_utils.span_utils import to_bioul

def spans_to_bio_tags(spans, length) :
    tag_sequence = ['O'] * length
    for span in spans :
        start, end, label = span
        tag_sequence[start] = 'B-' + label
        for ix in range(start + 1, end) :
            tag_sequence[ix] = 'I-' + label

    return to_bioul(tag_sequence, encoding='BIO')