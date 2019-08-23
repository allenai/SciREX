import os
import re
import click

from tqdm import tqdm

BASEPATH = os.getenv("nfs_path", ".")
from scripts.entity_utils import *

BRAT_CONFIGS_FOLDER = os.path.join(BASEPATH, "scripts/brat_configs")

def add_to_text_with_ent(text, w, ent_id, enttype, ann_file) :
    start = len(text)
    text += w
    end = len(text)
    ann_file.write('T' + str(ent_id) + '\t' + enttype + ' ' + str(start) + ' ' + str(end) + '\t' + w + '\n')
    ent_id += 1
    return text, ent_id 


def generate_folders_for_documents(documents, BRAT_ANNO_FOLDER):
    for doc_id, row in documents.items():
        folder = os.path.join(BRAT_ANNO_FOLDER, doc_id)
        os.makedirs(folder, exist_ok=True)
        annotation_config = (
            "[entities]\n"
            + "\n".join(used_entities)
            + "\n"
            + "\n".join([x + "_Predicted" for x in used_entities])
            + "\n\n"
        )

        annotation_config += "\n" + "[attributes]\nSalient\tArg:<ENTITY>\n\n"
        annotation_config += "\n" + open(os.path.join(BRAT_CONFIGS_FOLDER, "relations.conf")).read()
        f = open(folder + "/annotation.conf", "w")
        f.write(annotation_config)
        f.close()

        visual_config = open(os.path.join(BRAT_CONFIGS_FOLDER, "drawing.conf")).read()
        visual_config += '\n\n' + '[labels]\n'
        f = open(folder + "/visual.conf", "w")
        f.write(visual_config)
        f.close()


def add_rows_to_files(document, txt_file, ann_file):
    text = ""
    ent_id = 1
    att_id = 1
    coref_id = 1
    start = 0
    words = document["words"]
    tokens = []
    paragraphs = document['paragraphs']
    ends = [x[1] for x in paragraphs]

    for i, tok in enumerate(words):
        if i in ends :
            para_end = "<PARAGRAPH_END>\n"
            start += len(para_end)
            text += para_end
            
        tokens.append({"start": start, "end": start + len(tok), "text": tok})
        start += len(tok) + 1
        text += tok + " "

    span_to_entid = {}
    span_to_enttype = {}

    for (s, e), label in document['prediction'].items():
        enttype = label.split("_")[1] + "_Predicted"
        linktype = label.split('_')[2]
        ann_file.write("T" + str(ent_id) + "\t")
        ann_file.write(enttype + " " + str(tokens[s]["start"]) + " " + str(tokens[e - 1]["end"]) + "\t")
        ann_file.write(text[tokens[s]["start"] : tokens[e - 1]["end"]] + "\n")
        if linktype == 'True' :
            ann_file.write('A' + str(att_id) + '\tSalient T' + str(ent_id) + '\n')
            att_id += 1
        ent_id += 1

    for (s, e), label in document['gold'].items():
        enttype = label.split("_")[1]
        ann_file.write("T" + str(ent_id) + "\t")
        ann_file.write(enttype + " " + str(tokens[s]["start"]) + " " + str(tokens[e - 1]["end"]) + "\t")
        ann_file.write(text[tokens[s]["start"] : tokens[e - 1]["end"]] + "\n")
        span_to_entid[(s, e)] = ent_id
        span_to_enttype[(s, e)] = enttype
        ent_id += 1

    # for (s1, e1), (s2, e2), p in document['coref_prediction'] :
    #     if p > 0.6 :
    #         ann_file.write('R' + str(coref_id) + '\t')
    #         ann_file.write('Coreference' + ' ARG1:T' + str(span_to_entid[(s1, e1)]) + ' ARG2:T' + str(span_to_entid[(s2, e2)]))
    #         ann_file.write('\n')
    #         coref_id += 1

    # for (s1, e1), (s2, e2), p in document['relation_scores'] :
    #     if (s1, e1) not in span_to_entid : continue
    #     if (s2, e2) not in span_to_entid : continue
    #     if s1 < s2 and p > document['relation_threshold'] and span_to_enttype[(s1, e1)] != span_to_enttype[(s2, e2)]:
    #         ann_file.write('R' + str(coref_id) + '\t')
    #         ann_file.write('Relation' + ' ARG1:T' + str(span_to_entid[(s1, e1)]) + ' ARG2:T' + str(span_to_entid[(s2, e2)]))
    #         ann_file.write('\n')
    #         coref_id += 1


    text +=  ('\n\n')
    text +=  ('GOLD LABELS\n\n')

    for k in document['n_ary_relations'][0].keys() :
        text, ent_id = add_to_text_with_ent(text, k, ent_id, k, ann_file)
        text += ' || '

    text +=  ('\n')
    for rel in document['n_ary_relations'] :
        text +=  (" || ".join([rel[k] for k in rel.keys()]) + '\n')

    text +=  ('\n\n')

    for k in document['n_ary_relations'][0].keys() :
        text, ent_id = add_to_text_with_ent(text, k, ent_id, k, ann_file)
        text +=  ('\n')
        ent = set([rel[k] for rel in document['n_ary_relations']])
        for e in ent :
            if e in document['gold_clusters'] :
                text +=  (e + ' || ' + ' | '.join(document['gold_clusters'][e]['words']))
                text +=  ('\n')

        text +=  ('\n')

    text +=  ("\n\n")
    text +=  ('MODEL OUTPUTS\n\n')
    clusters = document['clusters']
    for n, tuples in document['predicted_relations'].items() :
        for k, v in tuples.items() :
            for e in k :
                text, ent_id = add_to_text_with_ent(text, e, ent_id, e, ann_file)
                text += ' || '
            text +=  ('\n')
            for t in v :
                for e in t :
                    text +=  (clusters[e]['name'] + ' || ')
                    if n == 1 :
                        text +=  (' | '.join(list(clusters[e]['words'])))
                text +=  ('\n')
            text +=  ('\n' + "="*100 + '\n')

    txt_file.write(text)


def generate_brat_annotations(documents, brat_anno_folder):
    generate_folders_for_documents(documents, brat_anno_folder)
    for doc_id, document in tqdm(documents.items()):
        filename = os.path.join(brat_anno_folder, doc_id, "document")
        txt_file, ann_file = open(filename + ".txt", "w"), open(filename + ".ann", "w")
        add_rows_to_files(document, txt_file, ann_file)
        txt_file.close()
        ann_file.close()
