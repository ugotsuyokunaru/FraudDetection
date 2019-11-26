import gc

import numpy as np
import pandas as pd
import networkx as nx
from stellargraph.data import BiasedRandomWalk
from stellargraph import StellarGraph
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

from src.feature import get_combine

def construct_graph(col_a, col_b, df):
    G = nx.Graph()
    B = nx.Graph()
    for index, row in df.iterrows():
        if G.has_edge(row[col_a], row[col_b]):
            G[row[col_a]][row[col_b]]['weight'] += 1
        else:
            G.add_edge(row[col_a], row[col_b], weight=1)

    B.add_nodes_from(list(df[col_b].unique()))
    for i in df[col_b].unique():
        for tup in [(mid, G.neighbors(mid)) for mid in G.neighbors(i)]:
            for j in tup[1]:
                if(i != j):
                    w = (G[i][tup[0]]['weight'] + G[j][tup[0]]['weight']) / 2
                    if B.has_edge(i, j):
                        B[i][j]['weight'] += w
                    else:
                        B.add_edge(i, j, weight=w)
                else:
                    if(not B.has_edge(i, j)):
                        B.add_edge(i, j, weight=1)
    del G
    gc.collect()

    return B

def graph_embed():
    combine = get_combine()
    li = ['bank', 'acquirer', 'coin', 'mcc', 'shop', 'nation', 'city']
    d = {
        'bank': 'b',
        'mcc': 'm',
        'acquirer': 'a',
        'coin': 'c',
        'shop': 's',
        'nation': 'n',
        'city': 'z'
    }
    have_df = False
    df_all = None
    
    for col_a in li:
        combine[col_a] = combine[col_a].astype(str) + d[col_a]
    
    for index, col_a in enumerate(li[1:]):
        print(f'{col_a} started..')
        walk_all = []
        for day in np.linspace(1, 120, 120):
            print(day, end=',')
            df = combine[combine['date'] == day]
            G = construct_graph(col_a, 'bank', df)
            rw = BiasedRandomWalk(StellarGraph(G))
            walk = rw.run(
                nodes=list(G.nodes()), # root nodes
                length=80,  # maximum length of a random walk
                n=1,        # number of random walks per root node 
                p=1,       # Defines (unormalised) probability, 1/p, of returning to source node
                q=1,        # Defines (unormalised) probability, 1/q, for moving away from source node
            )
            walk_all.extend(walk)
            del df, G, rw, walk
            gc.collect()
            
        model = Word2Vec(walk_all, size=5, window=3, min_count=1, sg=0, workers=16, iter=10)
        temp_d = {}
        for w in list(model.wv.vocab):
            temp_d[w] = model[w]
        temp_df = pd.DataFrame(
            data = combine[col_a].map(temp_d).tolist(),
            columns = ['embed_bank_' + col_a + str(x + 1) for x in range(5)]
        )
        if(have_df):
            df_all = pd.concat([df_all, temp_df], axis=1)
        else:
            df_all = temp_df
            have_df = True
        del temp_d, model
        gc.collect()
    return df_all