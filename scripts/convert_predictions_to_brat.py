import os
import re
import click

from tqdm import tqdm

BASEPATH = os.getenv("nfs_path", ".")
from scripts.entity_utils import *

BRAT_CONFIGS_FOLDER = os.path.join(BASEPATH, "scripts/brat_configs")


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

        annotation_config += "\n" + open(os.path.join(BRAT_CONFIGS_FOLDER, "relations.conf")).read()
        f = open(folder + "/annotation.conf", "w")
        f.write(annotation_config)
        f.close()

        visual_config = open(os.path.join(BRAT_CONFIGS_FOLDER, "drawing.conf")).read()
        f = open(folder + "/visual.conf", "w")
        f.write(visual_config)
        f.close()


def add_rows_to_files(document, txt_file, ann_file):
    text = ""
    ent_id = 1
    coref_id = 1
    start = 0
    words = document["words"]
    tokens = []

    for tok in words:
        tokens.append({"start": start, "end": start + len(tok), "text": tok})
        start += len(tok) + 1
        text += tok + " "

    span_to_entid = {}

    for (s, e), label in document['prediction'].items():
        enttype = label.split("_")[1] + "_Predicted"
        ann_file.write("T" + str(ent_id) + "\t")
        ann_file.write(enttype + " " + str(tokens[s]["start"]) + " " + str(tokens[e - 1]["end"]) + "\t")
        ann_file.write(text[tokens[s]["start"] : tokens[e - 1]["end"]] + "\n")
        span_to_entid[(s, e)] = ent_id
        ent_id += 1

    for (s, e), label in document['gold'].items():
        enttype = label.split("_")[1]
        ann_file.write("T" + str(ent_id) + "\t")
        ann_file.write(enttype + " " + str(tokens[s]["start"]) + " " + str(tokens[e - 1]["end"]) + "\t")
        ann_file.write(text[tokens[s]["start"] : tokens[e - 1]["end"]] + "\n")
        ent_id += 1

    # for (s1, e1), (s2, e2), p in document['coref_prediction'] :
    #     if p > 0.6 :
    #         ann_file.write('R' + str(coref_id) + '\t')
    #         ann_file.write('Coreference' + ' ARG1:T' + str(span_to_entid[(s1, e1)]) + ' ARG2:T' + str(span_to_entid[(s2, e2)]))
    #         ann_file.write('\n')
    #         coref_id += 1

    txt_file.write(text)


# @click.command()
# @click.option('--pwc_doc_file')
# @click.option('--pwc_sentence_file')
# @click.option('--pwc_prediction_file')
# @click.option('--brat_anno_folder')
def generate_brat_annotations(documents, brat_anno_folder):
    generate_folders_for_documents(documents, brat_anno_folder)
    for doc_id, document in tqdm(documents.items()):
        filename = os.path.join(brat_anno_folder, doc_id, "document")
        txt_file, ann_file = open(filename + ".txt", "w"), open(filename + ".ann", "w")
        add_rows_to_files(document, txt_file, ann_file)
        txt_file.close()
        ann_file.close()
