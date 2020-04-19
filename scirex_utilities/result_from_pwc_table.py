from typing import List, Tuple, NamedTuple
import glob
import csv

   
tables_dirname = '/home/sarthakj/data/arxiv/papers/'

class Result(NamedTuple):
    score: float
    table_name: str
    matching_clusters: List[List[str]]
    matching_cluster_items: List[str]


def get_scores(arxiv_id: str, clusters: List[List[str]]) ->  List[Result]:
    results = []
    arxiv_id_prefix = arxiv_id.split('.')[0]
    for filename in glob.glob(f'{tables_dirname}/{arxiv_id_prefix}/{arxiv_id}/table_*.csv'):
        print(filename)
        row_headars = {}
        col_headers = {}
        vals = []

        with open(filename) as f:
            csv_reader = csv.reader(f, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    for i, h in enumerate(row):
                        row_headars[h] = i
                else:
                    col_headers[row[0]] = line_count
                vals.append(row)
                line_count += 1
    
        matching_clusters = []

        for cluster in clusters:
            matching_span_row = None
            matching_span_col = None
            for span in cluster:
                if span in row_headars:
                    matching_span_row = span
                if span in col_headers:
                    matching_span_col = span
            
            if matching_span_row is not None or matching_span_col is not None:
                matching_clusters.append((cluster, matching_span_row, matching_span_col))
        
        for row_cluster in matching_clusters:
            row_header = row_cluster[1]
            if row_header is None:  # not matchin one of the headers in the first row
                continue
            for col_cluster in matching_clusters:
                col_header = col_cluster[2]
                if col_header is None:  # not matching one of the headers in the first col
                    continue
                score = vals[col_headers[col_header]][row_headars[row_header]]
                result = Result(score, filename, [row_cluster[0], col_cluster[0]], [row_header, col_header])
                results.append(result)
            
    return results


if __name__ == "__main__":
    '''  # table 2
    Method,F1,Acc.
    logistic reg.,0.7282,0.8108
    LDA diag.,0.7332,0.7872
    LDA full,0.7352,0.8141
    QDA diag.,0.7290,0.7979
    QDA full,0.7266,0.8108
    '''  
    results = get_scores('1202.1568', [['f1', 'F1', 'F1 score'], ['en', 'English'], ['LDA full', 'LDA-full'], ['lstm', 'long short term memory']])
    for result in results: 
        print(f'headers: {", ".join(result.matching_cluster_items)} ==> Score: {result.score}')