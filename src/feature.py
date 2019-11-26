import pandas as pd
import numpy as np 

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

# def trade_ratio(X, b, c):
#     # b grouped by c
#     dct = (X.groupby([c, b]).size() / X.groupby(c).size()).to_dict()
#     X[b + '_' + c + '_traderatio']  = X.set_index([c, b]).index.map(dct.get)
#     return None

class FeatureEngineer():
    def __init__(self):
        '''
        fe_dict: dict of arrays, record groups of engineering feature for drop columns use.
        '''
        self.fe_dict = {}
        return None
    
    def get_hour(self, df):
        df['time'] = df['time'].astype(int).astype(str).str.zfill(6)
        df['hour'] = df['time'].str.slice(0, 2).astype(int)
        df['time'] = df['time'].astype('int64')
        return None

    def numerical_stat(self, df, num_col, columns, \
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
    
    def num_agg(self, df, a, b):
        '''
        one col a has how many col b. (ex. 一個帳號有幾張卡)
        p.s. suitable for categorical to categorical
        '''
        df[b + '_num/' + a] = df[a].map(df.loc[:, [a, b]].groupby(a).nunique().loc[:, b].to_dict())
        return None

    def trade_ratio(self, df, a, b):
        '''
        b trade ratio (ex. 貨幣在那個國家交易過幾次)
        p.s. suitable for categorical to categorical
        '''
        dct = (df.groupby([a, b]).size() / df.groupby(a).size()).to_dict()
        df[b + '_trade_ratio/' + a] = df.set_index([a, b]).index.map(dct.get)
        return None

    def sum_money(self, df):
        df['sum_money'] = (df.groupby('card'))['money'].cumsum()
        return None

    def diff_money(self, df):
        df['diff_money'] = abs((df.groupby('card'))['money'].diff())

    def card_first_trade_sum(self, df):
        df['card_first_trad'] = 0
        a = df.reset_index().groupby(['card']).first()['index']
        df.loc[a, 'card_first_trad'] = 1
        df['sum_unicard'] = (df.groupby('bank'))['card_first_trad'].cumsum()


    # def coin_bank_dominance(self, df):
    #     '''
    #     dominant coin type 貨幣在那個歸戶交易過幾次
    #     '''
    #     dct = (df.groupby(['bank','coin']).size() / df.groupby('bank').size()).to_dict()
    #     df['coin_bank_dominance'] = df.set_index(['bank', 'coin']).index.map(dct.get)
    #     return None

    # def money_divide_term(self, df):
    #     '''
    #     money / term => term_money 分期金額
    #     '''
    #     df['money_term'] = df['money'] / (df['term'] + 1)
    #     return None

    # def count_city_num(self, df):
    #     '''
    #     一個國家有幾個城市
    #     '''
    #     df['city_num'] = df['nation'].map(df.loc[:, ['city', 'nation']].groupby('nation').count().loc[:,'city'].to_dict())
    #     return None

    # def categorical_dummy_ratio(self, df, cats, dummies):
    #     '''
    #     dummy variable for each categorical feature trade ratio
    #     ex. nation / acquirer / bank, fallback / online / 3ds / install / excess 的交易數量比例
        
    #     cats: list for categorical features
    #     dummies: list for dummy features
    #     '''
    #     if(not isinstance(cats, list)):
    #         raise ValueError('cats params need to be a list')
    #     if(not isinstance(dummies, list)):
    #         raise ValueError('dummies params need to be a list')
    #     for c in cats:
    #         for b in dummies:
    #             trade_ratio(df, b, c)
    #     return None

    # def week_day(self, df):
    #     '''
    #     week day of the trade
    #     '''
    #     df['week_day'] = df['date'] % 7
    #     return None


    def engineer_all(self, df):
        agg = ['acquirer', 'bank', 'card', 'coin', 'mcc', 'shop', 'city', 'nation', 'status', 'trade_cat', 'pay_type', 'trade_type']
        need_encode = ['acquirer', 'bank', 'card', 'coin', 'mcc', 'shop', 'city', 'nation']
        num = 7

        # number aggregation
        print('=' * 40, '\n[1/{}] num agg ...\n'.format(num)+'=' * 40)
        before_fe = df.columns.shape[0]
        for col in need_encode:
            li_temp = [x for x in agg if x != col]
            for b in li_temp:
                if(((col == 'city') and (b == 'nation')) or ((col == 'card') and (b == 'bank'))):
                    pass
                else:
                    print(f'one {col} has how many {b}')
                    self.num_agg(df, col, b)
        self.fe_dict['num_agg'] = list(df.columns[before_fe:])

        # ratio aggregation
        ratio = ['acquirer', 'bank', 'card', 'coin', 'mcc', 'shop', 'city', 'nation', \
            'status', 'trade_cat', 'pay_type', 'trade_type', 'fallback', '3ds', 'online', 'install', 'excess']
        print('\n' + '=' * 40, '\n[2/{}] trade ratio ...\n'.format(num) +'=' * 40)
        before_fe = df.columns.shape[0]
        for col in need_encode:
            li_temp = [x for x in ratio if x != col]
            for b in li_temp:
                if(((col == 'city') and (b == 'nation')) or ((col == 'card') and (b == 'bank'))):
                    pass
                else:
                    print(f'{b} trade ratio for one {col}')
                    self.trade_ratio(df, col, b)
        self.fe_dict['trade_ratio'] = list(df.columns[before_fe:])

        # count by day
        print('\n\n' + '=' * 40, '\n[3/{}] count by day ...\n'.format(num) +'=' * 40)
        before_fe = df.columns.shape[0]
        for col in need_encode:
            print(col, end=',')
            df[col + '_count_1'] = df.set_index(['date', col]).index.map(df.groupby(['date', col]).size().to_dict())
        self.fe_dict['count_one_day'] = list(df.columns[before_fe:])

        # money stat feature
        print('\n\n' + '=' * 40, '\n[4/{}] money stat ...\n'.format(num) +'=' * 40)
        before_fe = df.columns.shape[0]
        self.numerical_stat(df, 'money', need_encode, var=False, mean=False, min_val=False)
        self.fe_dict['money_stat'] = list(df.columns[before_fe:])

        # sum money
        print('\n\n' + '=' * 40, '\n[5/{}] sum money ...\n'.format(num) +'=' * 40)
        self.sum_money(df)

        # diff money
        print('\n\n' + '=' * 40, '\n[6/{}] diff money ...\n'.format(num) +'=' * 40)
        self.diff_money(df)

        # card first trade sum
        print('\n\n' + '=' * 40, '\n[7/{}] card first trade sum ...\n'.format(num) +'=' * 40)
        self.card_first_trade_sum(df)

        print('\nDONE')
        return None