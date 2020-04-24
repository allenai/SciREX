import os, json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)

def make_sciERC_into_pwc_format(instance, put_coref=False):
    doc_id = instance['doc_key']
    words = [w for s in instance["sentences"] for w in s]
    sections = [[0, len(words)]]
    sentences = [[0, len(words)]]

    ner = [(s, e + 1, 'Generic' if put_coref else l) for sentence in instance['ner'] for (s, e, l) in sentence]

    existing_entities = set([(s, e) for sentence in instance['ner'] for (s, e, l) in sentence])
    coref = {str(i) : [(s, e + 1) for (s, e) in cluster] for i, cluster in enumerate(instance['clusters'])}

    for cluster in instance['clusters'] :
        for span in cluster :
            if tuple(span) not in existing_entities :
                ner.append((span[0], span[1] + 1, 'Generic'))

    n_ary_relations = []

    return {
        'doc_id' : doc_id,
        'words' : words,
        'sections' : sections,
        'sentences' : sentences,
        'ner' : ner,
        'coref' : coref if put_coref else {},
        'n_ary_relations' : n_ary_relations
    }
    

def dump_sciERC_to_file(input_dir, output_dir):
    os.makedirs(output_dir + '/ner_version', exist_ok=True)
    os.makedirs(output_dir + '/coref_version', exist_ok=True)

    for split in ["train", "dev", "test"]:
        data = [json.loads(line) for line in open(os.path.join(input_dir, split + ".json"))]
        data = [make_sciERC_into_pwc_format(ins, put_coref=True) for ins in data]
        f = open(os.path.join(output_dir + '/coref_version', split + ".jsonl"), "w")
        f.write("\n".join([json.dumps(ins) for ins in data]))
        f.close()

    for split in ["train", "dev", "test"]:
        data = [json.loads(line) for line in open(os.path.join(input_dir, split + ".json"))]
        data = [make_sciERC_into_pwc_format(ins, put_coref=False) for ins in data]
        f = open(os.path.join(output_dir + '/ner_version', split + ".jsonl"), "w")
        f.write("\n".join([json.dumps(ins) for ins in data]))
        f.close()

if __name__ == '__main__' :
    args = parser.parse_args()
    dump_sciERC_to_file(args.input_dir, args.output_dir)