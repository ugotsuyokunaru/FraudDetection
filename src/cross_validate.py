import gc

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt 
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit


def lgb_f1_score(y_hat, data, THRESHOLD=0.248):
    y_true = data.get_label()
    y_hat = np.where(y_hat >= THRESHOLD, 1, 0)
    return 'f1', f1_score(y_true, y_hat), True

def lgb_train(train_data, val_data, threshold, init_model, boost_round=1000, random_seed=6, for_submit=False):
    print('boost round: ', boost_round)
    def lgb_f1_score(y_hat, data, THRESHOLD=threshold):
        y_true = data.get_label()
        y_hat = np.where(y_hat >= THRESHOLD, 1, 0)
        return 'f1', f1_score(y_true, y_hat), True

    valid_sets = [train_data] if for_submit else [train_data, val_data]

    params = {
        'objective': 'binary',
        # 'early_stopping_rounds': 100,
        'learning_rate': 0.01,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'max_depth': -1,
        'num_leaves': 100,
        'seed': random_seed,
        'metrics': 'None'
    }
    eval_dict = {}
    clf = lgb.train(params, 
        train_data,
        valid_sets=valid_sets,
        evals_result=eval_dict,
        num_boost_round=boost_round,
        verbose_eval=100,
        init_model=init_model,
        feval=lgb_f1_score)

    if for_submit:
        del eval_dict
        gc.collect()
        return clf
    else:
        lgb.plot_metric(eval_dict, metric='f1')
        res = max(eval_dict['valid_1']['f1'])
        del eval_dict
        gc.collect()
        return res

class CrossValidate():
    def __init__(self, threshold=0.248):
        self.threshold = threshold
    
    def sliding_window(self, X, y, cat, boost_round=500, n_fold=3, random_seed=6):
        '''
        This is the implementation of sliding window cross validation.
        ==================================
        | train| test |      |      |
        |      | train| test |      |
        |      |      | train| test |
        ==================================
        X: train data
        y: train label
        cat: categorical list
        '''
        shape_fold = int(X.shape[0] / (n_fold + 1))
        shape_list = []
        for i in range(n_fold):
            shape_list.append(shape_fold * (i + 1))
        Xs = np.split(X, shape_list)
        ys = np.split(y, shape_list)
        
        res = []
        for i in range(n_fold):
            train_data = lgb.Dataset(data=Xs[i], label=ys[i], categorical_feature=cat)
            val_data = lgb.Dataset(data=Xs[i], label=ys[i], reference=train_data, categorical_feature=cat)
            res.append(lgb_train(train_data, val_data, threshold=self.threshold, init_model=None, boost_round=boost_round, random_seed=random_seed))
            del train_data, val_data
            gc.collect()
        del Xs, ys
        gc.collect()
        return res
    
    def expanding_window(self, X, y, cat, boost_round=500, n_fold=3, random_seed=6):
        '''
        This is the implementation of expanding window cross validation.
        ==================================
        | train| test |      |      |
        |   train     | test |      |
        |       train        | test |
        ==================================
        X: train data
        y: train label
        cat: categorical list
        last_fold: the last folds want to run (> 0)
        '''
        print(f'threshold: {self.threshold}')
        tscv = TimeSeriesSplit(n_fold)
        res = []
        count = 1
        for train_index, val_index in tscv.split(X):
            print('\n' + '='*40 + '\n' + '[ CV Round {} ]'.format(count))
            count += 1
            X_tr, X_val = X.iloc[train_index, :], X.iloc[val_index, :]
            y_tr, y_val = y[train_index], y[val_index]
            train_data = lgb.Dataset(data=X_tr, label=y_tr, categorical_feature=cat)
            val_data = lgb.Dataset(data=X_val, label=y_val, reference=train_data, categorical_feature=cat)
            res.append(lgb_train(train_data, val_data, threshold=self.threshold, init_model=None, boost_round=boost_round, random_seed=random_seed))
            del train_data, val_data
            gc.collect()
            print('\n' + '='*40 + '\n')
        return res