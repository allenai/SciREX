import re
from bs4 import BeautifulSoup
import bs4
from bs4 import Comment


import os
in_dir = '../../../beltagy/result_extraction/data/xmls'
out_dir = 'data/plain_text'

def clean_p(elem):
    texts = []
    for content in elem.contents:
        if isinstance(content, bs4.element.NavigableString):
            texts.append(content)
        elif isinstance(content, bs4.element.Tag):
            if content.name in ['text', 'emph', 'verbatim', 'sup']:
                texts.append(content.text)
            elif content.name in ['ref'] :
                if content.has_attr('labelref') :
                    texts.append('<pwc_cite>' + content['labelref'] + '</pwc_cite>')
            elif content.name in ['break', 'note', 'item', 'cite', 'ref', 'error', 'math', 'graphics']:
                pass
            elif content.name in ['tag', 'inline-block', 'tabular', 'rule', 'xmtok', 'xmapp', 'toccaption',
                                  'caption', 'inline-enumerate', 'xmarray', 'g']:
                pass  # TODO
            else:
                print(f'UNKNOWWWN tag: {str(content)[:60]} =================================================')
        else:
            print(f'UNKNOWWWN class: {str(content)[:60]} =================================================')
    return " ".join(texts)

for filename in os.listdir(in_dir):
    if filename.endswith(".xml"):
        in_file = os.path.join(in_dir, filename)
        out_file = os.path.join(out_dir, filename)
        print(in_file, out_file)
        try:
            with open(in_file) as f:
                doc = f.read()
            with open(out_file, 'w') as f:
                soup = BeautifulSoup(doc)
                comments = soup.find_all(string=lambda text: isinstance(text, Comment))
                for c in comments :
                    c.extract()
                for elem in soup.find_all(re.compile('title|^p$')):
                    if elem.name == 'p':
                        f.write(clean_p(elem))
                        f.write('\n')
                    elif elem.name == 'title':
                        f.write('Latex Section Start\n\n')
                        f.write(f'{elem.parent.name}: {clean_p(elem)}')
                        f.write('<Latex Title End>')
                        f.write('\n')
        except:
            print(f'Parsing doc failed: {in_file} <><><><><><><><><><><><><><><><><>')
        
            