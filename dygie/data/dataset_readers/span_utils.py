from allennlp.data.dataset_readers.dataset_utils.span_utils import to_bioul

def spans_to_bio_tags(spans, length) :
    tag_sequence = ['O'] * length
    for span in spans :
        start, end, label = span
        tag_sequence[start] = 'B-' + label
        for ix in range(start + 1, end) :
            tag_sequence[ix] = 'I-' + label

    return to_bioul(tag_sequence, encoding='BIO')

def generate_seq_field(span_dict, length, element_map) :
    tag_sequence = [None]*length
    for (start, end), element in span_dict.items() :
        for j in range(start, end) :
            tag_sequence[j] = element

    tag_sequence = [element_map(x) for x in tag_sequence]
    return tag_sequence
        
