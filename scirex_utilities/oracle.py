import json

no_body_text = 0
total = 0
score_found_count = 0
metric_found_count = 0
task_found_count = 0
method_found_count = 0
dataset_found_count = 0
all_found_count = 0
with open('pwc_s2.jsonl') as f:
    for line in f:
        d = json.loads(line)
        model_name = d['model_name']
        dataset = d['dataset']
        metric = d['metric']
        score = d['score']
        task = d['task']
        body_text = d['s2_body_text'] + d['s2_abstract'] + d['s2_title']
        found = True
        if str(score) in body_text:
            score_found_count += 1
        else:
            found = False
        if metric in body_text:
            metric_found_count += 1
        else:
            found = False
        if task in body_text:
            task_found_count += 1
        else:
            found = False
        if model_name in body_text:
            method_found_count += 1
        else:
            found = False
        if dataset in body_text:
            dataset_found_count += 1
        else:
            found = False

        if found:
            all_found_count += 1
                    
        total += 1

print('total', total)
print('score', score_found_count)
print('metric', metric_found_count)
print('task', task_found_count)
print('method', method_found_count)
print('dataset', dataset_found_count)
print('all', all_found_count)


