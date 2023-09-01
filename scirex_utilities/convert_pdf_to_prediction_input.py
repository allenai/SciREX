from argparse import ArgumentParser

from scirex_utilities.io_util import *
from scirex_utilities.preprocessing.add_cleaned_text_to_pwc import *
from scirex_utilities.preprocessing.pdf_parser import *
from scirex_utilities.preprocessing.process_pwc_to_sentences import *


def convert_to_prediction_input(doc_id, json_sent):
    cur_section_id = -1
    words = []
    sentences = []
    sections = []
    for sent_info in json_sent:
        offset = len(words)
        if cur_section_id != sent_info['section_id']:
            cur_section_id = sent_info['section_id']
            sections.append([offset, offset])
        tokens = sent_info['sentence'].split()
        words.extend(tokens)
        sentences.append([offset, len(words)])
        sections[-1][1] = len(words)
    return {"doc_id": doc_id, "words": words, "sentences": sentences, "sections": sections}


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    doc_id = args.paper_id

    print("Processing paper", doc_id)
    print("--------------------------")
    # 1- parse grobid text
    print("parse by grobid ..")
    parse_by_grobid(input_dir, doc_id, output_dir)

    # 2- extract text from grobid
    print("Process the output from grobid")
    grobid_text = read_grobid_file(input_dir, doc_id)

    # 3- convert text to json
    print("convert to sentences ...")
    json_sent = convert_to_sentences(doc_id, grobid_text)

    # 4- convert to prediction input
    print("convert sentences to predition inout")
    prediction_input = convert_to_prediction_input(doc_id, json_sent)

    # 5- write output
    write_json(join(output_dir, str(doc_id) + "_scirex_prediction_input.json"), prediction_input, indent=0)


if __name__ == '__main__':
    parser = ArgumentParser("Convert a pdf paper to the prediction input formate.")
    parser.add_argument("input_dir", help="The input directory of the paper .")
    parser.add_argument("output_dir", help="The output directory of the result.")
    parser.add_argument("paper_id", help="The paper id, the pdf name should be the same.")

    args = parser.parse_args()
    main(args)
