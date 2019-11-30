import os
import gc
from argparse import ArgumentParser

import pickle as pkl
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.cross_validate import CrossValidate, lgb_train
from src.threshold import Threshold
from src.models import train_lgb

TRAIN_SHAPE = 1521787

def get_dataset(d_type='120days'):
    ''' get engineered features dataframe from pickle file
    Args:
        d_type (str) : dataframe type (120days / 30days)
    '''
    file_path = './data/combine_{}.pkl'.format(d_type)
    with open(file_path, 'rb') as f:
        dataset = pkl.load(f)
    return dataset

def full_train_predict(combine, cat, not_train, threshold, init_model=None, boost_round=1000):
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
    submit.to_csv('./submit/david.csv', index=False)
    print('\nPrediction written to ./submit/david.csv')
    
    del X, y, train_data, test, submit
    gc.collect()

    return clf

def train_30days():
    '''
    Prep: df_train_1_30, df_train_31_60, df_train_61_90, df_test (All after Feature Engineering + embedding)

    Intro: After feature engineering, you should get df_train splited by three parts:
        [df_train_1_30, df_train_31_60, df_train_61_90], and df_test. We then concat 
        the three datasets back and name it as 'combine'.
    '''
    drop_list = ['acquirer','bank','card','coin','mcc','shop','city','nation','date', 'txkey', 'fraud_ind', 'time']

    # 1. load preprocess features
    print('\n[Step 1/3] Load features after Feature Engineer Pre-processing ... \n')
    combine = get_dataset(d_type='30days')
    df_test = combine.loc[TRAIN_SHAPE:]
    # Training data preperation
    y_train = combine[['fraud_ind']]
    X_train = combine.drop(columns=drop_list)
    # Testing data preperation
    test_txkey = df_test[['txkey']]
    df_test = df_test.drop(columns=drop_list)
    print('\tTrain dataset shape :', X_train.shape)
    print('\tTrain label shape :', y_train.shape, '\n')

    # 2. Training Focal model
    print('\n[Step 2/3] Start Training Focal model ... \n')
    train_lgb(X_train, y_train, df_test, test_txkey, model_type='focal')

    # 3. Training diff model
    print('\n[Step 3/3] Start Training diff model ... \n')
    train_lgb(X_train, y_train, df_test, test_txkey, model_type='diff')

def train_120days(action='cv', n_fold=5, threshold=None):
    not_train = ['txkey', 'date', 'time', 'fraud_ind']
    need_encode = ['acquirer', 'bank', 'card', 'coin', 'mcc', 'shop', 'city', 'nation']
    cat = ['status', 'trade_cat', 'pay_type', 'trade_type', 'hour']
    feature_root = os.path.join('.', 'data', 'feature')
    os.makedirs(feature_root, exist_ok=True)

    # 1. load preprocess features
    print('\n[Step 1/3] Load features after Feature Engineer Pre-processing ... \n')
    dataset = get_dataset(d_type='120days')
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
        full_train_predict(dataset, cat, not_train + need_encode, threshold=best_threshold)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_type", "-m", choices=['30', '120', 'both'], default='both', type=str)
    parser.add_argument("--action", "-a", choices=['cv', 'submit'], default='submit', type=str)
    parser.add_argument("--n_fold", "-n", default=5, type=int)
    parser.add_argument("--threshold", "-t", default=0.26, type=float)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = parse_args()
    if opt.model_type == 'both':
        print('[Start training model 1 (120 days) ]\n')
        train_120days(
            action=opt.action,
            n_fold=opt.n_fold,
            threshold=opt.threshold
        )
        print('[Start training model 2 3 (30 days) ]\n')
        train_30days()
    elif opt.model_type == '120':
        print('[Start training (120 days) model ]\n')
        train_120days(
            action=opt.action,
            n_fold=opt.n_fold,
            threshold=opt.threshold
        )
    elif opt.model_type == '30':
        print('[Start training (30 days) model ]\n')
        train_30days()