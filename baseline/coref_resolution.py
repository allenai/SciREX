from baseline.baseline import compute_features, get_all_span_pair_features
from scripts.convert_brat_annotations_to_json import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import classification_report
import warnings

import random

def sample(spans, k=500, generate_all=False):
    pos = []
    neg = []
    if generate_all :
        k = float('inf')
    i = 1
    for s1 in spans:
        for s2 in spans:
            if s1 == s2:
                continue
            if len(s1.links) == 0 and len(s2.links) == 0:
                continue
            if len(s1.links & s2.links) != 0:
                if i <= k:
                    pos.append((s1, s2))
                elif random.random() < k / i:
                    pos[random.randrange(k)] = (s1, s2)
                i += 1

    n = min(k, max(10, len(pos))) if not generate_all else float('inf')
    i = 1
    for s1 in spans:
        for s2 in spans:
            if s1 == s2:
                continue
            if len(s1.links) == 0 and len(s2.links) == 0:
                continue
            if len(s1.links & s2.links) == 0:
                if i <= n:
                    neg.append((s1, s2))
                elif random.random() < n / i or (len(s1.links) > 0 and len(s2.links) > 0):
                    neg[random.randrange(n)] = (s1, s2)
                i += 1

    return {"pos": pos, "neg": neg}


def combine_mention_by_type(mlist):
    mlist = [y for x in mlist for y in x]
    mentions_by_enttype = {k: [] for k in used_entities}
    for m in mlist:
        mentions_by_enttype[m.entity].append(m)
    return mentions_by_enttype


def train_model(df_concat, end_id):
    df_concat = df_concat[df_concat.doc_id < end_id]
    fclusters = df_concat.groupby("doc_id").progress_apply(compute_features)
    fclusters = fclusters.apply(combine_mention_by_type)
    fclusters = fclusters.apply(pd.Series)
    samples = fclusters[used_entities].progress_applymap(lambda x: sample(x, generate_all=True))

    samples_done = samples[samples.index < end_id]
    doc_split = {}
    doc_split["train"], doc_split["test"] = train_test_split(samples_done.index, test_size=0.2)

    for t in used_entities:
        pos_feat, neg_feat = {}, {}
        pos_sample, neg_sample = {}, {}
        for split in doc_split:
            pos_sample[split] = [
                x
                for y in list(samples_done[samples_done.index.isin(doc_split[split])][t].apply(lambda x: x["pos"]))
                for x in y
            ]
            neg_sample[split] = [
                x
                for y in list(samples_done[samples_done.index.isin(doc_split[split])][t].apply(lambda x: x["neg"]))
                for x in y
            ]

            pos_feat[split] = list(map(lambda x: get_all_span_pair_features(*x), pos_sample[split]))
            neg_feat[split] = list(map(lambda x: get_all_span_pair_features(*x), neg_sample[split]))

        vec = DictVectorizer(sparse=False)
        vec.fit(pos_feat["train"] + neg_feat["train"])

        X, y = {}, {}
        for split in doc_split:
            X[split] = vec.transform(pos_feat[split] + neg_feat[split])
            y[split] = np.array([1] * len(pos_feat[split]) + [0] * len(neg_feat[split]))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            lr = LogisticRegressionCV(class_weight="balanced", penalty="l2", scoring="accuracy", cv=5)
            lr.fit(X["train"], y["train"])

        pred = lr.predict(X["test"])
        wrong_ix = [i for i in range(len(pred)) if pred[i] != y["test"][i]]
        wrong_ix = np.random.choice(wrong_ix, size=10)

        print(t)
        print(classification_report(y['train'], lr.predict(X["train"])))
        print(classification_report(y["test"], pred))
        print()
        print(sorted(list(zip(list(vec.vocabulary_.keys()), lr.coef_[0])), key=lambda x: x[1]))
        print()
        print([(x[0].text, x[1].text) for x in [(pos_sample["test"] + neg_sample["test"])[i] for i in wrong_ix]])
        print("=" * 200)

