import os

import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from scipy.misc import derivative

def focal_loss_lgb(y_pred, dtrain, alpha, gamma):
    a,g = alpha, gamma
    y_true = dtrain.label
    def fl(x,t):
        p = 1/(1+np.exp(-x))
        return -( a*t + (1-a)*(1-t) ) * (( 1 - ( t*p + (1-t)*(1-p)) )**g) * ( t*np.log(p)+(1-t)*np.log(1-p) )
    partial_fl = lambda x: fl(x, y_true)
    grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
    hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)
    return grad, hess

def sigmoid(x):
    return 1./(1. +  np.exp(-x))

def focal_loss_lgb_f1_score(preds, lgbDataset):
    preds = sigmoid(preds)
    binary_preds = [int(p>0.5) for p in preds]
    y_true = lgbDataset.get_label()
    return 'f1', f1_score(y_true, binary_preds), True

def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat), True

def train_lgb(X_train, y_train, df_test, test_txkey, model_type='focal'):   
    params = {
        'objective': 'binary',
        'early_stopping_rounds': 100,
        'learning_rate': 0.01,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'max_depth': -1,
        'num_leaves': 150,
        'seed': 44,
        'metric': ['auc', 'binary_logloss']
    }
    cat = ['trade_cat', 'trade_type', 'status', 'pay_type']
    train_data = lgb.Dataset(data=X_train, label=y_train, categorical_feature=cat)
    eval_dict = {}
    
    if model_type == 'focal':
        col_name = 'focal'
        thresh = 0.00001
        focal_loss = lambda x,y: focal_loss_lgb(x, y, alpha=0.5, gamma=1.)
        bst = lgb.train(
            params, 
            train_data,
            valid_sets=[train_data], 
            fobj=focal_loss,
            evals_result=eval_dict,
            num_boost_round=3000,
            verbose_eval=50,
            feval=focal_loss_lgb_f1_score
        )
    elif model_type == 'diff':
        col_name = 'lgbm_diff'
        thresh = 0.18
        bst = lgb.train(
            params, 
            train_data,
            valid_sets=[train_data],
            evals_result=eval_dict,
            num_boost_round=3000,
            verbose_eval=50,
            feval=lgb_f1_score
        )

    # Save Prediction to y_pred_focal
    df_pred = bst.predict(df_test)
    df_pred_prod = df_pred.copy()
    df_pred_prod[df_pred_prod > thresh] = 1
    df_pred_prod[df_pred_prod <= thresh] = 0
    # Create focal dataset
    df_output = test_txkey.copy()
    df_output[col_name] = df_pred_prod

    # write predicted result to csv file
    os.makedirs('./submit', exist_ok=True)
    df_output.to_csv('./submit/{}.csv'.format(model_type), index=False)
    print('\nPrediction written to ./submit/{}.csv'.format(model_type))