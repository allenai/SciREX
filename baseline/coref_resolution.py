from baseline.baseline import get_all_span_pair_features, Mention
from scripts.convert_brat_annotations_to_json import *
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import classification_report
import warnings
from typing import List, Tuple
from tqdm import tqdm

import random
import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.DEBUG)

def generate_pairs(file_path) -> List[Tuple[str, str, str]]:
    pairs = []
    with open(file_path, "r") as snli_file:
        i = 0
        for line in snli_file:
            ins = json.loads(line)
            entities = [(x[0], x[1]) for x in ins["ner"]]
            coref = {}
            for k, vlist in ins["coref"].items():
                for v in vlist:
                    if tuple(v) not in coref:
                        coref[tuple(v)] = []
                    coref[tuple(v)].append(k)
            coref = {k: set(v) for k, v in coref.items()}
            for e1, e2 in combinations(entities, 2):
                c1, c2 = coref.get(e1, set()), coref.get(e2, set())
                w1, w2 = " ".join(ins["words"][e1[0] : e1[1]]), " ".join(ins["words"][e2[0] : e2[1]])
                if w1.lower() == w2.lower() or len(c1 & c2) > 0:
                    gold_label = 1
                elif len(c1) == 0 and len(c2) == 0:
                    gold_label = "-"
                elif len(c1 & c2) == 0:
                    gold_label = 0

                if gold_label == "-":
                    continue

                pairs.append(
                    {
                        0: Mention(start=e1[0], end=e1[1], text=w1, mention_pos=0),
                        1: Mention(start=e2[0], end=e2[1], text=w2, mention_pos=0),
                        "y": gold_label,
                        "doc_id" : ins['doc_id']
                    }
                )

    return pairs


def generate_data(train_path, dev_path, test_path):
    logging.info("Starting ...")
    pairs = {}
    pairs["train"] = generate_pairs(train_path)
    pairs["dev"] = generate_pairs(dev_path)
    pairs["test"] = generate_pairs(test_path)

    logging.info("Done loading data ")

    for k, plist in pairs.items():
        for p in tqdm(plist) :
            p["features"] = get_all_span_pair_features(p[0], p[1])

    logging.info("Generate Features ...")
    vec = DictVectorizer(sparse=False)
    vec.fit([p["features"] for p in pairs["train"]])

    X, y = {}, {}
    for split, plist in pairs.items():
        X[split] = vec.transform([p["features"] for p in plist])
        y[split] = np.array([p["y"] for p in plist])

    logging.info("Vectorized Features ..")

    return vec, X, y, pairs

def fit_lr(vec, X, y) :
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        lr = LogisticRegressionCV(class_weight="balanced", penalty="l2", scoring="f1", cv=5)
        lr.fit(X["train"], y["train"])

    logging.info("Fitted Logistic Regression ..")

    pred = lr.predict(X["dev"])

    print(classification_report(y["train"], lr.predict(X["train"])))
    print(classification_report(y["dev"], pred))
    print()
    print(sorted(list(zip(list(vec.vocabulary_.keys()), lr.coef_[0])), key=lambda x: x[1]))
    print()
    print("=" * 200)

