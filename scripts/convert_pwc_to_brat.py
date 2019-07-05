import os
import re
import click

from tqdm import tqdm
BASEPATH = os.getenv('RESULT_EXTRACTION_BASEPATH', '.')
from scripts.analyse_pwc_entity_results import *

BRAT_CONFIGS_FOLDER = os.path.join(BASEPATH, 'scripts/brat_configs')

def generate_folders_for_documents(pwc_grouped, BRAT_ANNO_FOLDER) :
    for index, row in tqdm(pwc_grouped.iterrows()) :
        os.makedirs(BRAT_ANNO_FOLDER + row['s2_paper_id'], exist_ok=True)
        annotation_config = '[entities]\n' + '\n'.join(used_entities) + '\n\n'
        annotation_config += '[attributes]\n'
        visual_config = '[labels]\n'
        for k, v in map_true_entity_to_available.items() :
            annotation_config += "\n".join([re.sub(r'[^\w-]', '_', x) + '\tArg:' + v for x in row[k]]) + '\n'
            visual_config += '\n'.join([re.sub(r'[^\w-]', '_', x) + ' | ' + x for x in row[k]]) + '\n'

        annotation_config += '\n' + open(os.path.join(BRAT_CONFIGS_FOLDER, 'relations.conf')).read()
        f = open(BRAT_ANNO_FOLDER + row['s2_paper_id'] + '/annotation.conf', 'w')
        f.write(annotation_config)
        f.close()

        visual_config += '\n' + open(os.path.join(BRAT_CONFIGS_FOLDER, 'drawing.conf')).read()
        f = open(BRAT_ANNO_FOLDER + row['s2_paper_id'] + '/visual.conf', 'w')
        f.write(visual_config)
        f.close()

def add_rows_to_files(df, txt_file, ann_file) :
    text = ''
    ent_id = 1
    start = 0
    att_id = 1
    for index, row in df.iterrows():
        words = row['words'] + ['\n']
        tokens = []

        for tok in words :
            tokens.append({
                'start' : start,
                'end' : start + len(tok),
                'text' : tok
            })
            start += len(tok) + 1
            text += tok + ' '

        for enttype in used_entities :
            for tok_start, tok_end, _ in row[enttype] :
                ann_file.write('T' + str(ent_id) + '\t')
                ann_file.write(enttype + ' ' + str(tokens[tok_start]['start']) + ' ' + str(tokens[tok_end - 1]['end']) + '\t')
                ann_file.write(text[tokens[tok_start]['start']:tokens[tok_end - 1]['end']] + '\n')
                
                matched_true_entities = match_entity_with_best_truth(enttype, 
                                                                     text[tokens[tok_start]['start']:tokens[tok_end - 1]['end']],
                                                                     row[map_available_entity_to_true[enttype]])
                for match in matched_true_entities :
                    ann_file.write('A' + str(att_id) + '\t')
                    ann_file.write(re.sub(r'[^\w-]', '_', match) + ' T' + str(ent_id) + '\n')
                    att_id += 1
                ent_id += 1

    txt_file.write(text)

@click.command()
@click.option('--pwc_doc_file')
@click.option('--pwc_sentence_file')
@click.option('--pwc_prediction_file')
@click.option('--brat_anno_folder')
def generate_brat_annotations(pwc_doc_file, pwc_sentence_file, pwc_prediction_file, brat_anno_folder) :
    pwc_df = load_pwc_full_text(pwc_doc_file)
    pwc_grouped = pwc_df.groupby('s2_paper_id')[['dataset', 'task', 'model_name', 'metric']] \
                        .aggregate(lambda x : list(set(tuple(x)))).reset_index()

    pwc_sentences = load_pwc_sentence_predictions(pwc_sentence_file, pwc_prediction_file)

    pwc_sentences = pwc_sentences.merge(pwc_grouped, left_on='doc_id', right_on='s2_paper_id')
    pwc_sentences = pwc_sentences.sort_values(by=['doc_id', 'section_id', 'para_id', 'sentence_id']).reset_index(drop=True)

    pwc_sentences_grouped = pwc_sentences.groupby('doc_id')
    generate_folders_for_documents(pwc_grouped, brat_anno_folder)
    already_done = []
    for grp_name, df_group in tqdm(pwc_sentences_grouped) :
        if grp_name not in already_done :
            filename = brat_anno_folder + grp_name + '/document'
            txt_file, ann_file = open(filename + '.txt', 'w'), open(filename + '.ann', 'w')
            add_rows_to_files(df_group, txt_file, ann_file)
            txt_file.close()
            ann_file.close()

if __name__ == '__main__' :
    generate_brat_annotations()