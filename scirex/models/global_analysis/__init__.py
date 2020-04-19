import json
import numpy as np
import matplotlib.pyplot as plt
from scirex_utilities.entity_utils import *

def generate_matrix_for_document(document, span_field, matrix_field) :
    span2idx = {tuple(k):i for i, k in enumerate(document[span_field])}
    matrix = np.zeros((len(span2idx), len(span2idx)))
    for e1, e2, score in document[matrix_field] :
        matrix[span2idx[tuple(e1)], span2idx[tuple(e2)]] = score
        
    return matrix


