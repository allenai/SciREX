from allennlp.data.dataset_readers.dataset_utils.span_utils import to_bioul

def spans_to_bio_tags(spans, length) :
    tag_sequence = ['O'] * length
    for span in spans :
        is_inner_span = False
        for span_2 in spans :
            if (not is_same_span(span, span_2)) and is_x_in_y(span, span_2) :
                is_inner_span = True

        if is_inner_span :
            continue

        start, end, label = span
        tag_sequence[start] = 'B-' + label
        for ix in range(start + 1, end) :
            tag_sequence[ix] = 'I-' + label

    return to_bioul(tag_sequence, encoding='BIO')

is_same_span = lambda x, y : x[0] == y[0] and x[1] == y[1]
is_x_in_y = lambda x, y: x[0] >= y[0] and x[1] <= y[1]

does_overlap = lambda x, y: max(x[0], y[0]) < min(x[1], y[1])