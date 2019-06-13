import json
import elasticsearch
with open('data/evaluation-tables.json') as f:
    data = json.load(f)

corpus_size = 0
arxiv_counts = 0
unrolled_dicts = []

es = elasticsearch.Elasticsearch('http://es5.cf.production.s2.prod.ai2.in:9200')
not_found_in_es = 0
total_records = 0
no_paper_url = 0

def search_es_by_url(paper_url):
    results = es.search(index='paper', doc_type='paper',
                body={"query": {"match": {"sourceInfo.pdfUris": paper_url}}},
                _source_include=['bodyText', 'paperAbstract', 'title', 'id'], size=1000)

    if len(results['hits']['hits']) == 1:  # paper found in ES
        hit = results['hits']['hits'][0]
        return hit['_source']  # paper_record
    else:
        return None

def search_es_by_arxiv_url(paper_url):
    urls_to_try = [paper_url]
    input_paper_url = paper_url
    if 'arxiv' in paper_url and 'abs' in paper_url:
        paper_url = paper_url.replace('abs', 'pdf') + '.pdf'
        urls_to_try.append(paper_url)
    if 'http' in paper_url and 'https' not in paper_url:
        paper_url = paper_url.replace('http', 'https')
        urls_to_try.append(paper_url)
    if 'arxiv' in input_paper_url and 'abs' in input_paper_url and input_paper_url[-2] == 'v':
        paper_url = input_paper_url[:-2]
        urls_to_try.append(paper_url)
    if 'http' in paper_url and 'https' not in paper_url:
        paper_url = paper_url.replace('http', 'https')
        urls_to_try.append(paper_url)
    
    for paper_url in urls_to_try:
        paper_record = search_es_by_url(paper_url)
        if paper_record is not None:
            return paper_record
    return None

def print_task(cat_group):
    global corpus_size
    global arxiv_counts
    global unrolled_dicts
    global not_found_in_es
    global total_records
    global no_paper_url
    cat = cat_group['categories']
    task_name = cat_group['task']
    datasets = cat_group['datasets']
    if len(datasets) > 0:
        # print(cat, task_name, len(datasets))
        for dataset in datasets:
            dataset_name = dataset['dataset']
            sota = dataset['sota']
            rows = sota['rows']
            # print(dataset_name, len(rows), )
            corpus_size += len(rows)
            for row in rows:
                paper_url = row['paper_url']
                if 'arxiv' in paper_url:
                    arxiv_counts += 1
                if paper_url == "":
                    no_paper_url += 1

                paper_record = search_es_by_arxiv_url(paper_url)
                if paper_record is None:
                    paper_record = {'id': 'not_found', 'bodyText': '', 'title': '', 'paperAbstract': ''}
                    print(total_records, arxiv_counts, no_paper_url, not_found_in_es)
                    not_found_in_es += 1

                total_records += 1
                for metric, score in row['metrics'].items():
                    body_text = paper_record.get('bodyText') or ''
                    abstract = paper_record.get('paperAbstract') or ''
                    d = {'model_name': row['model_name'], 'metric': metric, 'score': score,
                            'paper_url': paper_url, 'task': task_name, 'category': cat,
                            'dataset': dataset_name,
                            'title': row['paper_title'],
                            's2_paper_id': paper_record['id'],
                            's2_body_text': body_text,
                            's2_abstract': abstract,
                            's2_title': paper_record['title'],
                        }
                    unrolled_dicts.append(d)

    for task in cat_group['subtasks']:
        print_task(task)

for cat_group in data:
    print_task(cat_group)

print(corpus_size)
print(arxiv_counts)
print(no_paper_url)
print(not_found_in_es)
with open('pwc_wp.jsonl', 'w') as f:
    for d in unrolled_dicts:
        s = json.dumps(d)
        f.write(s)
        f.write('\n')
