import pickle as pkl

import numpy as np
import pandas as pd

from src.feature import engineer_all, get_combine
from src.embed_targetcol import embed_target
from src.graph_embed import graph_embed

def preprocess():
    print('Start Preprocessing \nStart Embedding features engineering [1/3] ... \n')
    combine = get_combine()
    for col in ['money', 'trade_type', 'online']:
        print('Generating embed_target [{}] ... '.format(col))
        combine = pd.concat([combine, embed_target(col, epoch_num=50)], axis=1)
    print('Generating graph_embed ...')
    combine = pd.concat([combine, graph_embed()], axis=1)

    print('Start normal features engineering (method 1 : 120days) [2/3] ... \n')
    drop_shape = combine.shape[1]
    combine = engineer_all(combine)
    pkl.dump(combine, open('./data/combine_120days.pkl', 'wb'), protocol=4)
    combine.drop(combine.iloc[:, drop_shape:].columns, axis=1, inplace=True)

    print('Start normal features engineering (method 2 : 30days) [3/3] ... \n')
    data_list = []
    for day in [0, 30, 60, 90]:
        df_engineered = engineer_all(
            combine.loc[(combine['date'] >= day + 1) & (combine['date'] <= day + 30), :].copy())
        data_list.append(df_engineered)
    pkl.dump(pd.concat(data_list), open('./data/combine_30days.pkl', 'wb'), protocol=4)

if __name__ == '__main__':
    preprocess()