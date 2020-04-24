import os
import re
import click

from tqdm import tqdm
from collections import defaultdict
from runtime_scirex_utilities.read_all_files import get_all_file_outputs


BASEPATH = os.getenv("RESULT_EXTRACTION_BASEPATH", ".")
from scirex_utilities.entity_utils import *
from scirex_utilities.entity_matching_algorithms import *
from scirex_utilities.analyse_pwc_entity_results import *
from dygie.data.dataset_readers.paragraph_utils import move_boundaries

BRAT_CONFIGS_FOLDER = os.path.join(BASEPATH, "scirex_utilities/brat_configs")


def generate_folders_for_documents(pwc_grouped, BRAT_ANNO_FOLDER):
    for _, row in tqdm(pwc_grouped.iterrows()):
        os.makedirs(BRAT_ANNO_FOLDER + row["s2_paper_id"], exist_ok=True)
        annotation_config = "[entities]\n" + "\n".join(used_entities) + "\n\n"
        annotation_config += "[attributes]\n"
        visual_config = "[labels]\n"

        annotation_config += "\n".join([re.sub(r"[^\w-]", "_", x) + "\tArg:<ENTITY>" for x in ['Canonical Name']]) + "\n"
        visual_config += "\n".join([re.sub(r"[^\w-]", "_", x) + " | " + x for x in ['Canonical Name']]) + "\n"

        entities = defaultdict(list)
        for k, v in map_true_entity_to_available.items():
            for e in sorted(row[k]) :
                entities[e].append(v)
                if k == 'model_name' :
                    for _, c in chunk_string(e) :
                        entities[c].append(v)
                        
            for i in range(6) :
                entities[v + 'E_' + str(i)].append(v)
                    
        annotation_config += "\n".join([re.sub(r"[^\w-]", "_", x) + "\tArg:" + '|'.join(list(set(v))) for x, v in entities.items()]) + "\n"
        visual_config += "\n".join([re.sub(r"[^\w-]", "_", x) + " | " + x for x in entities]) + "\n"

        annotation_config += "\n" + open(os.path.join(BRAT_CONFIGS_FOLDER, "relations.conf")).read()
        f = open(BRAT_ANNO_FOLDER + row["s2_paper_id"] + "/annotation.conf", "w")
        f.write(annotation_config)
        f.close()

        visual_config += "\n" + open(os.path.join(BRAT_CONFIGS_FOLDER, "drawing.conf")).read()
        f = open(BRAT_ANNO_FOLDER + row["s2_paper_id"] + "/visual.conf", "w")
        f.write(visual_config)
        f.close()


def add_to_text(word, text, start):
    text += word + " "
    start += len(word) + 1
    return text, start


def add_rows_to_files(row, txt_file, ann_file):
    text = ""
    ent_id = 1
    start = 0
    att_id = 1

    first_row = row
    pwc_entities = {k:[] for k in true_entities}

    for v in true_entities:
        entities = first_row[v]
        text, start = add_to_text(map_true_entity_to_available[v].upper() + ":", text, start)
        for e in entities:
            e_start = start
            text, start = add_to_text(e, text, start)
            chunks = chunk_string(e)
            pwc_entities[v].append(e)
            for (s, ee), c in chunks :
                ann_file.write("T" + str(ent_id) + "\t")
                ann_file.write(map_true_entity_to_available[v] + " " + str(e_start + s) + " " + str(e_start + ee) + "\t")
                ann_file.write(c + "\n")
                ent_id += 1
                if v == 'model_name' :
                    pwc_entities[v].append(c)
            text, start = add_to_text("|", text, start)
            
        for i in range(6) :
            e = map_true_entity_to_available[v] + 'E_' + str(i)
            e_start = start
            text, start = add_to_text(e, text, start)
            
            ann_file.write("T" + str(ent_id) + "\t")
            ann_file.write(map_true_entity_to_available[v] + " " + str(e_start) + " " + str(start - 1) + "\t")
            ann_file.write(e + "\n")
            ent_id += 1
                
            text, start = add_to_text('|', text, start)

        text, start = add_to_text("\n", text, start)

    text, start = add_to_text("\n", text, start)

    elist = [x['span'] for x in row['combined']]
    plist = [x[1] - x[0] for x in row['sentence_limits']]
    plist = [list(x) for x in move_boundaries(plist, elist)]
    
    new_plist = []
    for i in range(len(plist)) :
        if plist[i][0] > plist[i][1] :
            plist[i+1][0] = plist[i][0]
            continue
        new_plist.append(plist[i])

    for p, q in zip(new_plist[:-1], new_plist[1:]) :
        assert p[1] == q[0]

    plist = new_plist
    tokens = []
    for s, e in sorted(plist):
        words = row["words"][s:e]
        for tok in words:
            tokens.append({"start": start, "end": start + len(tok), "text": tok})
            text, start = add_to_text(tok, text, start)

        text, start = add_to_text('\n', text, start)

    for span in row['combined']:
        tok_start, tok_end = span['span'][0], span['span'][1]
        enttype = span['label'].split('_')[1]
        ann_file.write("T" + str(ent_id) + "\t")
        ann_file.write(
            enttype + " " + str(tokens[tok_start]["start"]) + " " + str(tokens[tok_end - 1]["end"]) + "\t"
        )
        ann_file.write(text[tokens[tok_start]["start"] : tokens[tok_end - 1]["end"]] + "\n")

        written = text[tokens[tok_start]["start"] : tokens[tok_end - 1]["end"]]
        actual = " ".join(row['words'][tok_start:tok_end])
        if written != actual :
            print("ERROR", row['doc_id'], repr(written), actual)

        matched_true_entities = match_entity_with_best_truth(
            enttype,
            text[tokens[tok_start]["start"] : tokens[tok_end - 1]["end"]].replace('\n', ''),
            pwc_entities[map_available_entity_to_true[enttype]],
        )
        if len(matched_true_entities) > 0 :
            for match in matched_true_entities[:1]:
                ann_file.write("A" + str(att_id) + "\t")
                ann_file.write(re.sub(r"[^\w-]", "_", match) + " T" + str(ent_id) + "\n")
                att_id += 1
        ent_id += 1

    txt_file.write(text)


def generate_brat_annotations(
    pwc_doc_file, pwc_sentence_file, pwc_prediction_file, brat_anno_folder, after_doc_id=None
):
    pwc_df = load_pwc_full_text(pwc_doc_file)
    pwc_grouped = (
        pwc_df.groupby("s2_paper_id")[["dataset", "task", "model_name", "metric"]]
        .aggregate(lambda x: list(set(tuple(x))))
        .reset_index()
    )

    pwc_sentences = load_pwc_sentence_predictions(pwc_sentence_file, pwc_prediction_file)
    pwc_sentences = pwc_sentences.sort_values(by=["doc_id", "section_id", "para_id", "sentence_id"]).reset_index(
        drop=True
    )

    def combine(rows) :
        rows = rows.sort_values(by=['section_id', 'para_id', 'sentence_id']).to_dict(orient='records')
        words = []
        sentence_lims = []
        
        for row in rows :
            sentence_lims.append([len(words), len(words) + len(row['words'])])
            words += row['words']
            
        return pd.Series({'words' : words, 'sentence_limits' : sentence_lims})
    
    pwc_sentences_grouped = pwc_sentences.groupby('doc_id').apply(combine).reset_index()
    predicted_outputs = pd.DataFrame(get_all_file_outputs('../outputs/unannotated_results_folder/').values())

    pwc_sentences_grouped = pwc_sentences_grouped.merge(predicted_outputs, on='doc_id', how='inner')
    assert (pwc_sentences_grouped['words_x'] != pwc_sentences_grouped['words_y']).sum() == 0, breakpoint()
    pwc_sentences_grouped['words'] = pwc_sentences_grouped['words_x']
    pwc_sentences_grouped.drop(columns=['words_x', 'words_y'], inplace=True)

    pwc_sentences_grouped = pwc_sentences_grouped.merge(pwc_grouped, left_on="doc_id", right_on="s2_paper_id").to_dict(orient='records')

    generate_folders_for_documents(pwc_grouped, brat_anno_folder)
    for row in tqdm(pwc_sentences_grouped):
        grp_name = row['doc_id']
        if after_doc_id is None or row['doc_id'] > after_doc_id:
            filename = brat_anno_folder + grp_name + "/document"
            txt_file, ann_file = open(filename + ".txt", "w"), open(filename + ".ann", "w")
            add_rows_to_files(row, txt_file, ann_file)
            txt_file.close()
            ann_file.close()
