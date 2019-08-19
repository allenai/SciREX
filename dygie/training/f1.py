"""
Function to compute F1 scores.
"""


def safe_div(num, denom, m=100):
    if denom > 0 and num >= 0:
        return num*m / denom
    else:
        return -1


def compute_f1(predicted, gold, matched):
    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall, m=1)
    return precision, recall, f1
