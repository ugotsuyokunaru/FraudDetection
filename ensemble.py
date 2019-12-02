import gc
import os

import numpy as np
import pandas as pd

from train import get_dataset, TRAIN_SHAPE


def get_samelist():
    print('Loading features dataframe to generate "same_list" ... ')
    # make same user list, diff user list
    combine = get_dataset(d_type='30days')
    df_test = combine.loc[TRAIN_SHAPE:]
    same_list = list(set(combine['card']) & set(df_test['card']))
    del combine, df_test
    gc.collect()
    return same_list

def ensemble_helper(row, same_list):
    # ensemble logic
    if row['focal'] == 1:
        val = 1
    elif (row['lgbm_diff'] == 1) & (row['txkey'] not in same_list):  # different user
        val = 1
    elif (row['david'] == 1) & (row['txkey'] in same_list):          # same user
        val = 1
    else:
        val = 0
    return val

def ensemble():
    '''
    Prep:
        submission (the file that the platform provide for submit), 
        best model from david (take it from david), focal, lgbm_diff
    Intro: 
        merge all predictions
    '''
    print("\nLoading models' output csv files for ensembling ... ")
    submission = pd.read_csv('./submit/example.csv')
    focal = pd.read_csv('./submit/focal.csv')
    lgbm_diff = pd.read_csv('./submit/diff.csv')
    david = pd.read_csv('./submit/david.csv')
    submission = submission.merge(focal, on='txkey')
    submission = submission.merge(lgbm_diff, on='txkey', how='outer')
    submission = submission.merge(david, on='txkey', how='outer')
    submission.columns = ['txkey', 'fraud_ind', 'focal', 'lgbm_diff', 'david']

    print("\nStart Ensembling ... ")
    same_list = get_samelist()
    submission['fraud_ind'] = submission.apply(ensemble_helper, axis=1, args=(same_list))
    submission_ready = submission.drop(columns=['focal', 'lgbm_diff', 'david'])
    submission_ready.to_csv('./submit/ensemble.csv', index=False)

    print('\nFinal Prediction after ensembling written to ./submit/ensemble.csv')

if __name__ == '__main__':
    ensemble()