import pandas as pd
from sklearn.model_selection import train_test_split
import click

@click.command()
@click.option("--input_file")
def split_dataset(input_file) :
    df = pd.read_json(input_file, lines=True)
    df = df[df.s2_paper_id != 'not_found']
    ids = list(df.s2_paper_id.unique())
    train_ids, test_ids = train_test_split(ids, test_size=0.2, random_state=1)
    train_ids, dev_ids = train_test_split(train_ids, test_size=0.125, random_state=1)
    f = open('data/train_doc_ids.txt', 'w')
    f.write('\n'.join(train_ids))
    f.close()
    
    f = open('data/dev_doc_ids.txt', 'w')
    f.write('\n'.join(dev_ids))
    f.close()
    
    f = open('data/test_doc_ids.txt', 'w')
    f.write('\n'.join(test_ids))
    f.close()
    
    return len(train_ids)/len(ids), len(dev_ids)/len(ids), len(test_ids)/len(ids)

if __name__ == '__main__' :
    split_dataset()