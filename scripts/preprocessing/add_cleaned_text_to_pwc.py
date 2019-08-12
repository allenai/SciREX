from scripts.preprocessing.grobid_util import *

import os
BASEPATH = os.getenv('RESULT_EXTRACTION_BASEPATH', '.')

import pandas as pd
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
tqdm.pandas()

GROBID_OUTPUT_PATH = os.path.join(BASEPATH, '../../../beltagy/result_extraction/data/grobid/')
LATEX_OUTPUT_PATH = os.path.join(BASEPATH, 'data/plain_text/')
PWC_PATH = os.path.join(BASEPATH, '../../../beltagy/result_extraction/data/pwc_s2.jsonl')
PWC_CLEANED_OUTPUT_PATH = os.path.join(BASEPATH, 'data/pwc_s2_cleaned_text_v3.jsonl')

def get_arxiv_id(paper_url) :
    if 'arxiv' in paper_url and 'abs' in paper_url:
        return re.sub(r'.*arxiv\.org\/abs\/', '', paper_url)
    return None

def read_grobid_file(arxiv_id) :
    content = open(GROBID_OUTPUT_PATH + arxiv_id + '.tei.xml').read()
    soup = BeautifulSoup(content)
    
    text = soup.find('title').text
    if text is None :
        text = ''
    text += '\n\n'
    for section in soup.find_all('div') :
        section = extract_references_from_paragraph_text(section)
        text += '<title>section: ' + (section.name if section.name is not None else '') + '</title>\n'
        text += '\n'.join([re.sub(r'\s+', ' ', x).strip() for x in section.paragraphs])
        text += '\n\n'
        
    return text.strip()

def read_latex_file(arxiv_id) :
    content = open(LATEX_OUTPUT_PATH + arxiv_id + '.xml', errors='ignore').read()
    content = content.split('Latex Section Start')
    
    def clean(text) :
        text = re.sub(r'\n', ' ', text)
        text = text.replace(u'\xa0', u' ')
        text = re.sub(r'\\[^\s]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    cleaned_content = []
    for c in content :
        c = c.strip()
        if len(c) == 0 :continue
        if len(re.findall('Latex Title End', c)) > 1 :
            breakpoint()
        
        splits = c.split('<Latex Title End>')
        if len(splits) == 2 :
            title, text = splits
            title = clean(title.strip())
            text = text.strip()
        else :
            text = splits[0].strip()
            title = ''
        assert '\n\n' not in text
        cleaned_content.append('<title>' + title + '</title>\n' + text + '\n\n')
                
    return "".join(cleaned_content).strip()

def get_cleaned_text_if_available(row) :
    if row['arxiv_id'] is not None :
        grobid_text = read_grobid_file(row['arxiv_id']) if row['arxiv_id'] in grobid_outputs else ''
        if row['arxiv_id'] in latex_outputs :
            row['cleaned_text'] = read_latex_file(row['arxiv_id'])
            row['clean_type'] = 'latex'
            
            if len(row['cleaned_text']) > 500 and len(row['cleaned_text']) / max(1, len(grobid_text)) > 0.5:
                return row
            else :
                print("Using Grobid/Normal Text for ", row['arxiv_id'])
                
        if row['arxiv_id'] in grobid_outputs :
            row['cleaned_text'] = grobid_text
            row['clean_type'] = 'grobid'
            return row
    row['cleaned_text'] = row['s2_title'] + ' ' + row['s2_abstract'] + ' ' + row['s2_body_text']
    row['clean_type'] = 'normal'
    return row

if __name__ == '__main__' :
    grobid_outputs = os.listdir(GROBID_OUTPUT_PATH)
    grobid_outputs = [x.replace('.tei.xml', '') for x in grobid_outputs if 'tei' in x]

    latex_outputs = os.listdir(LATEX_OUTPUT_PATH)
    latex_outputs = [x.replace('.xml', '') for x in latex_outputs if 'xml' in x]

    pwc_df = pd.read_json(PWC_PATH, lines=True)
    pwc_df['arxiv_id'] = pwc_df['paper_url'].apply(get_arxiv_id)

    cleaned_pwc = pwc_df.drop_duplicates('s2_paper_id').progress_apply(get_cleaned_text_if_available, axis=1)
    pwc_df = pwc_df.merge(cleaned_pwc[['s2_paper_id', 'cleaned_text']], on='s2_paper_id')

    pwc_df = pwc_df.drop(columns=['cleaned_text'])
    pwc_df = pwc_df.merge(cleaned_pwc[['s2_paper_id', 'cleaned_text', 'clean_type']], on='s2_paper_id')

    pwc_df = pwc_df.drop(columns=['s2_title', 's2_abstract', 's2_body_text'])

    pwc_df.to_json(PWC_CLEANED_OUTPUT_PATH, orient='records', lines=True)
