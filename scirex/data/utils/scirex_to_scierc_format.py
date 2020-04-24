import argparse
import os
from itertools import combinations

from tqdm import tqdm

from scirex.data.dataset_readers.scirex_full_reader import ScirexFullReader
from scirex.data.utils.span_utils import is_x_in_y
from scirex_utilities.convert_brat_annotations_to_json import annotations_to_jsonl
from scirex_utilities.entity_utils import Relation, used_entities

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)

def convert_scirex_instance_to_scierc_format(instance) :
    words = instance['paragraph']
    sentence_indices = instance['sentence_indices']
    mentions = instance['ner_dict']
    start_ix, end_ix = instance['start_ix'], instance['end_ix']
    metadata = instance['document_metadata']

    instance_id = metadata['doc_id'] + ':' + str(instance['paragraph_num'])
    mentions = {(span[0] - start_ix, span[1] - start_ix): label for span, label in mentions.items()}
    sentence_indices = [(sent[0] - start_ix, sent[1] - start_ix) for sent in sentence_indices]

    ner = [[] for _ in range(len(sentence_indices))]
    for mention in mentions :
        in_sent = set([i for i, sent in enumerate(sentence_indices) if is_x_in_y(mention, sent) ])
        assert len(in_sent) == 1, breakpoint()
        ner[list(in_sent)[0]].append([mention[0], mention[1], mentions[mention][0]])

    sentences = [words[sent[0]:sent[1]] for sent in sentence_indices]
    span_to_cluster_ids = metadata['span_to_cluster_ids']
    num_clusters = len(metadata['cluster_name_to_id'])
    clusters = [[] for _ in range(num_clusters)]

    for span, cluster_ids in span_to_cluster_ids.items() :
        span = (span[0] - start_ix, span[0] - end_ix)
        if span in mentions and len(cluster_ids) > 0:
            clusters[cluster_ids[0]].append(span)

    relations = [[] for _ in range(len(sentence_indices))]

    for idx, sentence_mentions in enumerate(ner) :
        for span_1, span_2 in combinations(sentence_mentions, 2) :
            span_1_orig = (span_1[0] + start_ix, span_1[1] + start_ix)
            span_2_orig = (span_2[0] + start_ix, span_2[1] + start_ix)

            if span_1_orig in span_to_cluster_ids and span_2_orig in span_to_cluster_ids :
                ids_1 = span_to_cluster_ids[span_1_orig]
                ids_2 = span_to_cluster_ids[span_2_orig]
                if len(set(ids_1) & set(ids_2)) > 0:
                    relations[idx].append((span_1[0], span_1[1], span_2[0], span_2[1], 'USED_FOR'))

    ner = [[(int(s), int(e - 1), v) for (s, e, v) in sentence] for sentence in ner]
    clusters = [[(int(s), int(e-1)) for (s, e) in cluster] for cluster in clusters if len(cluster) > 0]
    relations = [[(int(s1), int(e1-1), int(s2), int(e2-1), l) for (s1, e1, s2, e2, l) in sentence] for sentence in relations]
    return {
        'doc_key' : instance_id,
        'ner' : ner,
        'sentences' : sentences,
        'clusters' : clusters,
        'relations' : relations
    }


def convert_scirex_to_scierc_format(file, output_file) :
    scirex_reader = ScirexFullReader(to_scirex_converter=True)._read(file)

    scierc_data = []
    for instance in tqdm(scirex_reader) :
        scierc_data.append(convert_scirex_instance_to_scierc_format(instance))

    num_relations = 0
    for instance in scierc_data :
        for sentence in instance['relations'] :
            num_relations += len(sentence)

    print(f"Writing to {output_file} {len(scierc_data)} instances")
    annotations_to_jsonl(scierc_data, output_file, 'doc_key')

if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    for key in ['train', 'dev', 'test'] :
        convert_scirex_to_scierc_format(
            os.path.join(args.input_dir, key + '.jsonl'), 
            os.path.join(args.output_dir, key + '.jsonl')
        )
