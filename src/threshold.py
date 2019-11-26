import gc

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.model_selection import TimeSeriesSplit

class Threshold():
    def __init__(self):
        pass

    def get_f1_score(self, threshold, y_true, y_proba):
        y_pred = np.where(y_proba >= threshold, 1, 0)
        return f1_score(y_true, y_pred)

    def lgb_f1_score_fixed(self, y_hat, data):
        y_true = data.get_label()
        y_hat = np.where(y_hat >= 0.5, 1, 0)
        return 'f1', f1_score(y_true, y_hat), True

    def threshold_search(self, y_true, y_proba):
        '''
        Using true label and probabily prediction from the model, 
        searching for the threshold taht maximize the f1_score

        Parameters: 
        y_true: true label
        y_proba: the probability from the model

        Return:
        search_result: dict of best threshold and it's best score
        '''
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        thresholds = np.append(thresholds, 1.001) 
        F = 2 / (1 / precision + 1 / recall) 
        best_score = np.max(F)
        best_th = thresholds[np.argmax(F)]
        search_result = {'threshold': best_th , 'f1': best_score}
        return search_result 

    def calc_threshold_diff(self, X, y, cat, n_fold, boost_round=1000):
        '''
        Use expanding window method to record each threshold difference from the fold's best f1 score.

        Parameters: 
        X: training data
        y: training label
        cat: categorical list for the lgb model
        n_fold: fold number
        boost_round: boosting round for the model

        Return:
        df: DataFrame with each threshold difference value from the fold's best f1 score.
        result_list: list of threshold on differrent fold
        '''
        tscv = TimeSeriesSplit(n_fold)
        params = {
            'objective': 'binary',
            # 'early_stopping_rounds': 100,
            'learning_rate': 0.01,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'max_depth': -1,
            'num_leaves': 100,
            'metric': 'None',
            'seed': 6
        }
        try_threshold = np.linspace(0.001, 0.999, 999)
        df = pd.DataFrame({'threshold': try_threshold})
        result_list = []
        i = 0
        for train_index, val_index in tscv.split(X):
            train_data = lgb.Dataset(data=X.loc[train_index, :], label=y[train_index], categorical_feature=cat)
            val_data = lgb.Dataset(data=X.loc[val_index, :], label=y[val_index], reference=train_data, categorical_feature=cat)
            clf = lgb.train(
                params, 
                train_data,
                valid_sets=[val_data],
                num_boost_round=boost_round,
                verbose_eval=100,
                feval=self.lgb_f1_score_fixed)
            # get best result f1 for val set
            prob = clf.predict(X.loc[val_index, :])
            search_result = self.threshold_search(y[val_index], prob)
            
            # add result
            i += 1
            df['fold_' + str(i)] = df['threshold'].apply(lambda x: self.get_f1_score(x, y[val_index], prob))
            df['fold_' + str(i)] = df['fold_' + str(i)] - search_result['f1']
            result_list.append(search_result)
            print(f'{i} fold run: search threshold {search_result}\n')
            
            # record max f1 score
            df.loc[999, 'fold_' + str(i)] = search_result['f1']
            del train_data, val_data, search_result
            gc.collect()
        return df, result_list

    def get_best_threshold(self, df):
        '''
        Within the range 0.05 from the min f1 score threshold, find the best threshold with the min std value
        
        Parameters:
        df: the DataFrame from calc_threshold_diff method

        Return:
        best_threshold: the best threshold value
        '''
        min_val = df.iloc[:999, :].set_index('threshold').mean(axis=1).argmax()
        best_threshold = df[(df['threshold'] >= min_val - 0.05) & (df['threshold'] <= min_val + 0.05)].set_index('threshold').std(axis=1).argmin()
        return best_threshold

    def get_cv_score(self, df, threshold):
        index = 1000 * (threshold - 0.001)
        return df.loc[[999, index], [x for x in df.columns if x != 'threshold']].sum().mean()