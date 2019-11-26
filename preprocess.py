import pickle as pkl

import numpy as np
import pandas as pd

from src.feature import engineer_all, get_combine
from src.embed_targetcol import embed_target
from src.graph_embed import graph_embed

def preprocess():
    combine = get_combine()
    for col in ['money', 'trade_type', 'online']:
        combine = pd.concat([combine, embed_target(col)], axis=1)
    combine = pd.concat([combine, graph_embed()], axis=1)

    drop_shape = combine.shape[1]
    combine = engineer_all(combine)
    pkl.dump(combine, open('./data/combine_120days.pkl', 'wb'))
    combine.drop(combine.iloc[:, drop_shape:].columns, axis=1, inplace=True)

    data_li = []
    for day in [0, 30, 60, 90]:
        data_li.append(engineer_all(
            combine.loc[
                (combine['date'] >= day + 1) & (combine['date'] <= day + 30), :]))
    pkl.dump(pd.concat([data_li]), open('./data/combine_30days.pkl', 'wb'))

if __name__ == '__main__':
    preprocess()