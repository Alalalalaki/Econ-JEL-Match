from collections import defaultdict
import requests
from xml.etree import ElementTree
import pandas as pd

def etree_to_dict(t):
    d = {t.tag: {} if t.attrib else None}
    children = list(t)
    if children:
        dd = defaultdict(list)
        for dc in map(etree_to_dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {t.tag: {k: v[0] if len(v) == 1 else v
                     for k, v in dd.items()}}
    if t.attrib:
        d[t.tag].update(('@' + k, v)
                        for k, v in t.attrib.items())
    if t.text:
        text = t.text.strip()
        if children or t.attrib:
            if text:
                d[t.tag]['#text'] = text
        else:
            d[t.tag] = text
    return d


def prepare_jel_description():
    response = requests.get("https://www.aeaweb.org/econlit/classificationTree.xml")
    tree = ElementTree.fromstring(response.content)
    d = etree_to_dict(tree)
    d3 = [[d2['classification'], d2["code"], d2["description"]]
          for d1 in d['data']['classification'] for d2 in d1['classification']]
    d3 = [[d_["code"], d_["description"], d2c, d2d]
          for d__, d2c, d2d in d3 for d_ in d__ if isinstance(d_, dict)]
    jel_des = pd.DataFrame(d3, columns=["jel", "des", "d2c", "d2d"])
    jel_des = jel_des.set_index("jel")
    return jel_des

def main():
    jel_des = prepare_jel_description()
    jel_des.to_pickle("../Data/jel_des.pkl")

if __name__ == '__main__':
    main()

