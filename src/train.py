import os
import gc
from argparse import ArgumentParser

import pickle as pkl
import numpy as np
import pandas as pd
import lightgbm as lgb

from feature import FeatureEngineer
from cross_validate import CrossValidate, lgb_train
from threshold import Threshold

def get_dataset(d_type='90day'):
    file_path = './data/combine_{}.pkl'.format(d_type)
    with open(file_path, 'rb') as f:
        dataset = pkl.load(f)
    return dataset

# def get_dataset():
#     li = ['acquirer', 'bank', 'card', 'money', 'trade_cat', 'coin', 'online', 'trade_type',\
#         'fallback', '3ds', 'fraud_ind', 'pay_type', 'install', 'term', 'date', 'time', 'mcc', 'shop', 'excess',\
#         'city', 'nation', 'status', 'txkey']

#     with open('./data/train_mr.pkl', 'rb') as f:
#         train = pkl.load(f)
#     with open('./data/test_mr.pkl', 'rb') as f:
#         test = pkl.load(f)

#     combine = pd.concat([train, test])
#     combine = combine.reset_index(drop=True)
#     combine = combine[li]   # reset dataframe column order (affect : LGBM sort by column index)
#     combine = combine.reset_index(drop=True)
#     return combine

def full_train_predict(combine, cat, not_train, threshold, file_name, init_model=None, boost_round=1000):
    TRAIN_SHAPE = 1521787

    X = combine.loc[:TRAIN_SHAPE - 1, [x for x in combine.columns if x not in not_train]]
    y = combine.loc[:TRAIN_SHAPE - 1, 'fraud_ind']
    print('X.shape :', X.shape)
    train_data = lgb.Dataset(data=X, label=y, categorical_feature=cat) 
    val_data = None

    # model training
    clf = lgb_train(train_data, val_data, threshold, init_model=init_model, boost_round=boost_round, for_submit=True)

    # predicting
    test = combine.loc[TRAIN_SHAPE:, [x for x in combine.columns if x not in not_train]]
    pred = clf.predict(test)
    submit = pd.DataFrame({
        'txkey': combine.loc[TRAIN_SHAPE:, 'txkey'],
        'fraud_ind': np.where(pred >= threshold, 1, 0)
    })

    # write predicted result to csv file
    os.makedirs('./submit', exist_ok=True)
    submit.to_csv(f'./submit/{file_name}.csv', index=False)
    del X, y, train_data, test, submit
    gc.collect()

    return clf

def train(action='cv', file_name='submit001', feature='new', feature_fname='feature_ver1l', n_fold=5, threshold=None):
    TRAIN_SHAPE = 1521787
    not_train = ['txkey', 'date', 'time', 'fraud_ind']
    need_encode = ['acquirer', 'bank', 'card', 'coin', 'mcc', 'shop', 'city', 'nation']
    cat = ['status', 'trade_cat', 'pay_type', 'trade_type']
    feature_root = os.path.join('.', 'data', 'feature')
    os.makedirs(feature_root, exist_ok=True)

    # 1. pre process
    print('\n[Step 1/3] Start Feature Engineer Pre-processing ... \n')
    feature_path = os.path.join(feature_root, feature_fname+'.pkl')
    if feature == 'new':
        # get dataset
        dataset = get_dataset()
        preprocessor = FeatureEngineer()
        preprocessor.engineer_all(dataset)
        with open(feature_path, 'wb') as file:
            pkl.dump(dataset, file)
    elif feature == 'load':
        with open(feature_path, 'rb') as file:
            dataset = pkl.load(file)
        print('Features loaded (from {})'.format(feature_path))
    # split train / test
    X = dataset.loc[:TRAIN_SHAPE - 1, [x for x in dataset.columns if x not in not_train and x not in need_encode]]
    y = dataset.loc[:TRAIN_SHAPE - 1, 'fraud_ind']
    print('\tTrain dataset shape :', X.shape)
    print('\tTrain label shape :', y.shape, '\n')

    # 2. calculate threshold
    print('\n[Step 2/3] Calculate best threshold ... \n')
    if threshold is None:
        th = Threshold()
        df, result_list = th.calc_threshold_diff(X, y, cat, n_fold=n_fold)
        best_threshold = th.get_best_threshold(df)
        for i, result in enumerate(result_list):
            print(f'\t {i} fold run: search threshold {result}')
    else:
        best_threshold = threshold
    print('\nBest Threshold (or given) = ', best_threshold)

    # 3. Training
    print('\n[Step 3/3] Start Training ... \n')
    if action == 'cv':
        cv = CrossValidate(threshold=best_threshold)
        res = cv.expanding_window(X, y, cat, boost_round=1000)
        print('>> Avg Cross Validation : {}'.format(sum(res) / len(res)))
    elif action == 'submit':
        full_train_predict(dataset, cat, not_train + need_encode, threshold=best_threshold, file_name=file_name)
        print('\nPrediction written to ./submit/{}.csv'.format(file_name))

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--action", "-a", choices=['cv', 'submit'], default='submit', type=str)
    parser.add_argument("--feature", "-f", choices=['new', 'load'], default='new', type=str)
    parser.add_argument("--feature_fname", "-fn", default='feature_ver1', type=str)
    parser.add_argument("--output_fname", "-on", default='submit001', type=str)
    parser.add_argument("--n_fold", "-n", default=5, type=int)
    parser.add_argument("--threshold", "-t", default=None, type=float)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = parse_args()

    train(
        action=opt.action,
        file_name=opt.output_fname,
        feature=opt.feature,
        feature_fname=opt.feature_fname,
        n_fold=opt.n_fold,
        threshold=opt.threshold
    )