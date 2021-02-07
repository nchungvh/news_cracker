import json
from networkx import json_graph
g = json_graph.node_link_graph(json.load(open('merge-G.json')))
with open('merge-G.csv','w',encoding = 'utf-8') as f:
    f.write('handle\ttext\tweight\n')
    for node in g.nodes(data=True):
        if node[1]['label'][0] == 1:
            continue
        neis = g.neighbors(node[0])
        for nei in neis:
            f.write('{}\t{}\t{:.2f}\n'.format(node[1]['content'][0], str(g.nodes[nei]['content'][0]), g[node[0]][nei]['weight']))
import pandas as pd
x = pd.read_csv('merge-G.csv',sep = '\t')
x = x.dropna()
x.to_csv('merge-G.csv',sep = '\t')
