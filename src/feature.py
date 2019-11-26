import pandas as pd
import numpy as np 


def get_combine():
    '''
    Read train.zip and test.zip files into dataframes. Rename these two dataframes 
    and combine them into one dataframe, finally replacing the null values in fallback and 3ds columns
    
    Return: 
        combine: a dataframe with train set and test set
    '''
    
    train = pd.read_csv('../data/train.zip', compression='zip')
    test = pd.read_csv('../data/test.zip', compression='zip')
    d = {
        'acqic': 'acquirer',
        'bacno': 'bank',
        'cano': 'card',
        'conam': 'money',
        'contp': 'trade_cat',
        'csmcu': 'coin',
        'ecfg': 'online',
        'etymd': 'trade_type',
        'flbmk': 'fallback',
        'flg_3dsmk': '3ds',
        'fraud_ind': 'fraud_ind',
        'hcefg': 'pay_type',
        'insfg': 'install',
        'iterm': 'term',
        'locdt': 'date',
        'loctm': 'time',
        'mcc': 'mcc',
        'mchno': 'shop',
        'ovrlt': 'excess',
        'scity': 'city',
        'stocn': 'nation',
        'stscd': 'status',
        'txkey': 'txkey'
    }
    d2 = {
        'Y': 1,
        'N': 0
    }
    train.rename(columns=d, inplace=True)
    test.rename(columns=d, inplace=True)
    combine = pd.concat([train.sort_values(by=['date', 'time']), test.sort_values(by=['date', 'time'])], sort=False).reset_index(drop=True)
    for col in ['fallback', '3ds', 'online', 'install', 'excess']:
        combine[col] = combine[col].map(d2)
    combine['fallback'].fillna(2, inplace=True)
    combine['3ds'].fillna(2, inplace=True)
    return combine

def map_stat_feature(X, b, c, mean=True, max_val=True, \
        min_val=True, std=True, var=True, median=True, median_diff=True):
    # b groupby c mean...
    if(mean):
        X[b + '_' + c + '_mean'] = \
            X[c].map(X.loc[:, [b, c]].groupby(c).mean().loc[:, b].to_dict())
    if(max_val):
        X[b + '_' + c + '_max'] = \
            X[c].map(X.loc[:, [b, c]].groupby(c).max().loc[:, b].to_dict())
    if(min_val):
        X[b + '_' + c + '_min'] = \
            X[c].map(X.loc[:, [b, c]].groupby(c).min().loc[:, b].to_dict())
    if(std):
        X[b + '_' + c + '_std'] = \
            X[c].map(X.loc[:, [b, c]].groupby(c).std().loc[:, b].to_dict())
    if(var):
        X[b + '_' + c + '_var'] = X[b + '_' + c + '_std'] ** 2
    if(median):
        X[b + '_' + c + '_median'] = \
            X[c].map(X.loc[:, [b, c]].groupby(c).median().loc[:, b].to_dict())
    if(median_diff):
        X[b + '_' + c + '_median_diff'] = \
            X[b] - X[b + '_' + c + '_median']
    return None
    
def get_hour(df):
    df['time'] = df['time'].astype(int).astype(str).str.zfill(6)
    df['hour'] = df['time'].str.slice(0, 2).astype(int)
    df['time'] = df['time'].astype('int64')
    return None

def numerical_stat(df, num_col, columns, \
    mean=True, max_val=True, min_val=True, std=True, var=True, median=True, median_diff=True):
    '''
    Mapping numerical feature to categorical feature, and get statistical value

    df: the dataframe you want to apply
    num_col: the numerical columns need to be groupby
    columns: list of columns want to interact with money
    mean: need mean inforamtion
    max: need max information
    std: need std information
    var: need var information
    median: need median information
    median_diff: need median differenc information
    '''
    if(not isinstance(columns, list)):
        raise ValueError('columns params need to be a list')
    for col in columns:
        map_stat_feature(df, num_col, col, mean=mean, max_val=max_val, min_val=min_val, std=std, \
            var=var, median=median, median_diff=median_diff)  
    return None

def count_by_bank(df, b):
    df[f'tradenum/bank-{b}'] = df.set_index(['bank', b]).index.map(df.groupby(['bank', b]).size())
    return None

def count_agg(df, a, b):
    df[f'tradenum/bank-{a}-{b}'] = df.set_index(['bank', a, b]).index.map(
        df.groupby(['bank', a, b]).size())
    return None

def num_agg(df, a, b):
    df[f'tradenum/{a}-{b}'] = df.set_index([a, b]).index.map(df.loc[:, [a, b]].groupby([a, b]).size().to_dict())
    return None

def oneday_count(df, col):
    df[f'tradenum/{col}-oneday'] = df.set_index(['date', col]).index.map(
        df.groupby(['date', col]).size().to_dict())
    return None

def sum_money(df, col):
    df[f'sum_money/{col}'] = (df.groupby(col))['money'].cumsum()
    return None

def engineer_all(df):
    need_encode = ['acquirer', 'bank', 'card', 'coin', 'mcc', 'shop', 'city', 'nation']

    get_hour(df)
    print('hour feature finished')

    need_count_bank = ['acquirer', 'coin', 'mcc', 'shop', 'city', 'nation', 'status', 'trade_cat', 'pay_type', 'trade_type', 'hour', \
                    'fallback', '3ds', 'online', 'install', 'excess']
    for col in need_count_bank:
        count_by_bank(df, col)
    print('aggregation feature from bank and other columns finished')

    count = ['fallback', '3ds', 'online', 'install', 'excess']
    need_count_agg = ['acquirer', 'coin', 'mcc', 'shop', 'city', 'nation', 'status', 'trade_cat', 'pay_type', 'trade_type', 'hour']
    for index, col in enumerate(need_count_agg):
        li_temp = count + need_count_agg[index + 1:]
        for b in li_temp:
            count_agg(df, col, b)
    print('aggregation feature from bank and other two columns finished')

    agg = ['fallback', '3ds', 'online', 'install', 'excess']
    need_count_agg = ['acquirer', 'card', 'coin', 'mcc', 'shop', 'city', 'nation', 'status', 'trade_cat', 'pay_type', 'trade_type', 'hour']
    for index, col in enumerate(need_count_agg):
        li_temp = agg + need_count_agg[index + 1:]
        for b in li_temp:
            num_agg(df, col, b)   
    print('aggregation feature from two columns finished')

    for col in need_encode:
        oneday_count(df, col)
    print('one day count for high cardinality columns finished')

    numerical_stat(df, 'money', need_encode, var=False, mean=False)
    print('money statistical features finished')

    for col in need_encode:
        sum_money(df, col)
    print('money cumulative sum feature finished')

    return df