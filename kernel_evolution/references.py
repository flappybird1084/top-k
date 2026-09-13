import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import urlopen


class Text(HTMLParser):
    def __init__(self):super().__init__();self.parts=[]
    def handle_data(self,data):self.parts.append(data.strip())


def fetch_references(root):
    root=Path(root)/'references';root.mkdir(parents=True,exist_ok=True)
    refs=[]
    for op in ['load','store','dot','sum','make_block_ptr']:
        url=f'https://triton-lang.org/main/python-api/generated/triton.language.{op}.html'
        path=root/(op+'.txt')
        if not path.exists():
            with urlopen(url,timeout=20) as response:html=response.read().decode()
            parser=Text();parser.feed(html)
            text=' '.join(x for x in parser.parts if x)
            start=text.find('triton.language.'+op+'(')
            path.write_text(text[max(0,start):max(0,start)+6000])
        refs.append({'url':url,'text':path.read_text()[:3500],'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    (root/'manifest.json').write_text(json.dumps(refs,indent=2))
    return refs
