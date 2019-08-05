import json
import re

import click
import pandas as pd
import spacy

nlp = spacy.load('en')

def convert_to_sentences(doc_id, text, clean_type) :
    json_sents = []
    sent_id = 0
    if clean_type in ['latex', 'grobid'] :
        # Divide into sections and then paragraphs
        text = [[y.strip() for y in x.strip().split('\n') if len(y.strip()) > 0] for x in text.split('\n\n') if len(x.strip()) > 0]
    else :
        text= [[text]]
    for section_id, paragraphs in enumerate(text) :
        for para_id, para in enumerate(paragraphs) :
            para = re.sub(r'<pwc_cite>(.*?)</pwc_cite>', '[reference]', para)
            if '<title>' in para and '</title>' in para:
                para = para.replace('<title>', '')
                para = para.replace('</title>', '')
                doc = nlp(para)
                # Do not divide title into sentences
                iterator = [doc]
            else :
                doc = nlp(para)
                iterator = doc.sents
            for sent_id, s in enumerate(iterator) :
                tokens = [t.text for t in s if len(t.text.strip()) != 0]
                s = " ".join(tokens)
                if len(s) == 0 :
                    continue
                json_sents.append({"sentence" : s, 
                                   "doc_id" : doc_id, 
                                   "sentence_id" : sent_id, 
                                   'para_id' : para_id, 
                                   'section_id' : section_id})
        
    return json_sents

@click.command()
@click.option("--input_file")
@click.option("--output_file")
def get_json_sentences(input_file, output_file) :
    pwc_df = pd.DataFrame(input_file, lines=True)[['s2_paper_id', 'paper_url', 'cleaned_text', 'clean_type']]
    pwc_df = pwc_df.drop_duplicates('s2_paper_id')

    def convert_row_to_sentences(row_tup) :
        index, row = row_tup
        json_sents = convert_to_sentences(row['s2_paper_id'], row['cleaned_text'], row['clean_type'])
        return json_sents

    from p_tqdm import p_map
    json_sents = []
    json_sents = p_map(convert_row_to_sentences, list(pwc_df.iterrows()), num_cpus=10)
    json_sents = [x for y in json_sents for x in y]

    with open(output_file, 'w') as f :
        for s in json_sents :
            f.write(json.dumps(s) + '\n')

if __name__ == "__main__":
    get_json_sentences()
