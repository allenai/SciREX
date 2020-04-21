import json
gold = [json.loads(line) for line in open('dygiepp/data/scierc/processed_data/json/test.json')]
predicted = [json.loads(line) for line in open('outputs/pwc_outputs/scirex_on_scierc_ner_outputs.jsonl')]

gold_ner = [[m for sentence in doc['ner'] for m in sentence] for doc in gold]
gold_ner = [list(map(tuple, doc)) for doc in gold_ner]

predicted_ner = [[(m['span'][0], m['span'][1] - 1, m['label']) for m in doc['predicted_ner']] for doc in predicted]
predicted_num = sum([len(s) for s in predicted_ner])
gold_num = sum([len(s) for s in gold_ner])

matched = sum([len(set(g) & set(p)) for g, p in zip(predicted_ner, gold_ner)])

p = matched / predicted_num
r = matched / gold_num
f1 = 2 * p * r / (p + r)

print(f"p = {p}")
print(f"r = {r}")
print(f"f1 = {f1}")