import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics
from sklearn.model_selection import train_test_split
import warnings
from sklearn.exceptions import ConvergenceWarning

def compute_paragraph_features(df_concat):
    def remove_words_in_entities(row):
        token_indices = [list(range(e.token_start, e.token_end)) for e in row["entities"] if len(e.links) > 0]
        token_indices = [x for y in token_indices for x in y]
        words = np.delete(np.array(row["words"]), token_indices)
        return words

    df_concat["filtered_words"] = df_concat[["words", "entities"]].progress_apply(remove_words_in_entities, axis=1)
    is_link_present = (
        df_concat.groupby(["doc_id", "para_num"])["entities"]
        .apply(lambda x: [e for l in x for e in l])
        .apply(lambda x: any([len(e.links) > 0 for e in x]))
        .reset_index()
    )

    words_features = (
        df_concat.groupby(["doc_id", "para_num"])["filtered_words"]
        .apply(lambda x: [e for l in x for e in l])
        .reset_index()
    )

    return is_link_present.merge(words_features, on=["doc_id", "para_num"])


def train_model(df_concat, end_id):
    para_features = compute_paragraph_features(df_concat)
    para_features = para_features[para_features.doc_id < end_id]
    para_features["paragraph"] = para_features["filtered_words"].apply(lambda x: " ".join(x).lower())

    doc_split = {}
    doc_split["train"], doc_split["test"] = train_test_split(para_features.doc_id.unique(), test_size=0.3)
    para_text, y = {}, {}
    for split in doc_split:
        df = para_features[para_features.doc_id.isin(doc_split[split])]
        para_text[split] = list(df["paragraph"])
        y[split] = np.array(df["entities"].apply(int))

    count = CountVectorizer(tokenizer=lambda x: x.split(), stop_words="english", min_df=3)
    count.fit(para_text["train"] + para_text["test"])

    X = {}
    for split in doc_split:
        X[split] = count.transform(para_text[split])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        lr = LogisticRegressionCV(class_weight="balanced", penalty="l2", scoring="accuracy", cv=3)
        lr.fit(X["train"], y["train"])

    pred = lr.predict_proba(X["test"])[:, 1] > 0.37
    print(metrics.classification_report(y["test"], pred))
    print("=" * 200)