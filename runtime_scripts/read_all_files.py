import json
import os
from tqdm import tqdm

from bisect import *


def get_all_file_outputs(output_dir):
    data = {}
    for d in tqdm(range(0, 1170, 20)):
        file = os.path.join(output_dir, str(d) + "_unannotated", "spans.jsonl")
        reads = [json.loads(line) for line in open(file)]
        for r in reads:
            data[r["doc_id"]] = r
            del data[r["doc_id"]]["gold"]
            del data[r["doc_id"]]["paragraphs"]

        file = os.path.join("../model_data/pwc_split_on_sectioned", str(d) + "_unannotated.jsonl")
        reads = [json.loads(line) for line in open(file)]
        for r in reads:
            data[r["doc_id"]]["ner"] = [
                {"span": [x[0], x[1]], "label": "Entity_" + x[2].split("_")[0]} for x in r["ner"]
            ]

    for k, v in data.items():
        predicted, gold = v["prediction"], v["ner"]
        predicted_to_add = []
        gs, ge = list(zip(*[(x["span"][0], x["span"][1]) for x in gold]))
        
        for p in predicted:
            s, e = p["span"]
            span_before = bisect_right(gs, s) - 1
            if span_before >= 0 and s < ge[span_before]:
                continue

            span_after = bisect_left(gs, s)
            if span_after < len(ge) and e > gs[span_after]:
                continue

            span_after = bisect_left(ge, e)
            if span_after < len(ge) and e > gs[span_after]:
                continue

            predicted_to_add.append(p)

        v["combined"] = sorted(gold + predicted_to_add, key=lambda x: x["span"][0])
        for p, q in zip(v["combined"][:-1], v["combined"][1:]):
            if p["span"][1] > q["span"][0]:
                breakpoint()

    return data

