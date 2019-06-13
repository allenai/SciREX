import pandas as pd
import json
pwc_data = [json.loads(line) for line in open('data/pwc_s2.jsonl')]
pwc_df = pd.DataFrame(pwc_data)[['s2_paper_id', 's2_title', 's2_abstract', 's2_body_text', 'paper_url']]
pwc_df = pwc_df.drop_duplicates('s2_paper_id')

import spacy
nlp = spacy.load('en')

def convert_to_sentences(doc_id, text_field, text) :
    json_sents = []
    sent_id = 0
    for sent_id, s in enumerate(nlp(text).sents) :
        tokens = [t.text for t in s if len(t.text.strip()) != 0]
        s = " ".join(tokens)
        if len(s) == 0 :
            continue
        json_sents.append({"sentence" : s, "doc_id" : doc_id, "text_field" : text_field, "sentence_id" : sent_id})
        
    return json_sents

def convert_row_to_sentences(row) :
    json_sents = []
    json_sents += convert_to_sentences(row['s2_paper_id'], 's2_title', row['s2_title'])
    json_sents += convert_to_sentences(row['s2_paper_id'], 's2_abstract', row['s2_abstract'])
    json_sents += convert_to_sentences(row['s2_paper_id'], 's2_body_text', row['s2_body_text'])
    return json_sents

from tqdm import tqdm
json_sents = []
for index, row in tqdm(pwc_df.iterrows(), total=pwc_df.shape[0]):
    json_sents += convert_row_to_sentences(row)

f = open('data/pwc_sentences.jsonl', 'w')
for s in json_sents :
    f.write(json.dumps(s) + '\n')
f.close()