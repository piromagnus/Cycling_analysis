from cv2 import mean, norm
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import StandardScaler
from enum import Enum

# create field type for heart rate, power and speed
class FieldType(Enum):
    HR = 1
    POWER = 2
    SPEED = 3


class PowerNormalTransformer(BaseEstimator, TransformerMixin):

    def __init__(self,
                 missing_value_threshold=0.75,
                 col_name='stream_watts',
                 col_time='stream_time',
                 method='ppr30',
                 max_ppr=1000,
                 rolling=False,
                 norm_data = None,
                 id_col = 'id',
                 keep_cols = []
                 ):
        self.col_name = col_name
        self.col_time = col_time
        self.id_col = id_col
        self.missing_value_threshold = missing_value_threshold
        self.method = method
        self.ppr = 0
        self.max_ppr = max_ppr
        self.roll = rolling 
        self.norm_data = norm_data

        self.keep_cols = keep_cols
    def fit(self, X):
        if self.roll:
            self.ppr = self.norm_data['ppr']
        else:
            self.ppr = X.groupby(self.id_col, group_keys=False)[self.col_name].rolling(30,min_periods=1).mean().max()

        self.X_ = X 
        return self

    def transform(self, X):
        check_is_fitted(self)
        index = X.index
        col = self.keep_cols+[self.id_col,self.col_time, self.col_name]
        tmp_X = X[col].copy()
        
        # print(tmp_X,self.ppr)
        # Normalizing the power with the ppr for each id
        if self.roll:
            tmp_X.loc[:, self.col_name] = tmp_X.groupby(self.id_col).apply(
                lambda x: x[self.col_name] / self.ppr[x.name]).reset_index()
        else:
            tmp_X.loc[:, self.col_name] = tmp_X[self.col_name] / self.ppr
        
        # print(tmp_X)
        # Delete the trainings with too many nans
        # tmp_X = delete_nans(tmp_X, self.missing_value_threshold, self.col_name, self.id_col)

        # Filling and Smoothing the power
        tmp_X = tmp_X.set_index([self.id_col, self.col_time])
        tmp_X = tmp_X.groupby(self.id_col, group_keys=False).apply(
            lambda x: x.fillna(0))
        tmp_X = tmp_X.groupby(self.id_col, group_keys=False).apply(
            lambda x: x.rolling(30,min_periods=1).mean()).reset_index()
        # tmp_X=tmp_X.set_index(index)
        return tmp_X


class HRNormalTransformer(BaseEstimator, TransformerMixin):

    def __init__(self,
                 missing_value_threshold=0.75,
                 col_name='stream_heartrate',
                 col_time='stream_time',
                 rolling=False,
                method='mean',
                norm_data = None,
                id_col = 'id',
                keep_cols = []
                 ):
        self.missing_value_threshold = missing_value_threshold
        self.col_name = col_name
        self.col_time = col_time
        self.id_col = id_col
        self.mean = 0
        self.std = 0
        self.roll = rolling 
        self.method = method
        self.norm_data = norm_data
        self.keep_cols = keep_cols

    def fit(self, X,):
        if self.roll:
            if self.method == 'mean':
                self.mean = self.norm_data["ma_hr"]
                self.std = self.norm_data["roll_std_hr"]
            else: 
                self.max = self.norm_data["roll_max_hr"]
        else:
            if self.method == 'mean':
                
                self.mean = X[self.col_name].mean()
                self.std = X[self.col_name].std()
            else:
                self.max = X[self.col_name].max()
        self.X_ = X
        return self

    def transform(self, X):
        check_is_fitted(self)

        index = X.index
        col = self.keep_cols+[self.id_col, self.col_time, self.col_name]
        tmp_X = X[col]
        # print(tmp_X.head())
        if self.roll:
            if self.method == 'mean':
                tmp_X.loc[:, self.col_name] = tmp_X.groupby(self.id_col).apply(
                    lambda x: (x[self.col_name] - self.mean.loc[x.name]) / (2 * self.std.loc[x.name])).reset_index()
            else:
                tmp_X.loc[:, self.col_name] = tmp_X.groupby(self.id_col).apply(
                    lambda x: x[self.col_name] / self.max.loc[x.name]).reset_index()
        else:
            if self.method == 'mean':
                tmp_X.loc[:, self.col_name] = (tmp_X[self.col_name] - self.mean) / (2 * self.std)
            else:
                tmp_X.loc[:, self.col_name] = tmp_X[self.col_name] / self.max

        # tmp_X = delete_nans(tmp_X, self.missing_value_threshold, self.col_name, self.id_col)

        # Filling and Smoothing the power
        tmp_X = tmp_X.set_index([self.id_col, self.col_time])
        tmp_X = tmp_X.groupby(self.id_col, group_keys=False).apply(
            lambda x: x.interpolate())
        tmp_X = tmp_X.groupby(self.id_col, group_keys=False).apply(
            lambda x: x.rolling(30,min_periods=1).mean()).reset_index()
        
        # tmp_X=tmp_X.set_index(index)
        return tmp_X


class AthleteTransformer(BaseEstimator, TransformerMixin):

    def __init__(self,
                 data_fields = ["stream_watts"],
                 col_time='stream_time',
                 id_col='id',
                 rolling=False,
                 hr_transformer=None,
                 power_transformer=None,
                 missing_value_threshold=0.50,
                 method_power='ppr30',
                 method_hr='mean',
                 max_ppr=1000,
                 norm_data = False,
                 keep_cols = []
                 ):

        self.missing_value_threshold = missing_value_threshold
        self.col_time = col_time
        self.id_col = id_col
        self.data_fields = data_fields
        self.method_power = method_power
        self.method_hr = method_hr
        self.max_ppr = max_ppr
        self.fields=[]
        self.roll = rolling
        self.norm_data = norm_data  
        self.keep_cols = keep_cols
        for i in data_fields:
            if i not in ['stream_heartrate', 'stream_watts', 'stream_speed']:
                raise ValueError('Data field {} not recognized'.format(i))
            #create a list of enum type
            if i == 'stream_heartrate':
                self.fields.append(FieldType.HR)
                if hr_transformer:
                    self.hr_transformer = hr_transformer
                else:
                    self.hr_transformer = HRNormalTransformer(missing_value_threshold=self.missing_value_threshold,
                                                            col_time=self.col_time,
                                                            col_name="stream_heartrate",
                                                            rolling=self.roll,
                                                            method= self.method_hr,
                                                            norm_data = self.norm_data,
                                                            id_col = self.id_col,
                                                            keep_cols = self.keep_cols
                                                            )
            elif i == 'stream_watts':
                self.fields.append(FieldType.POWER)
                if power_transformer:
                    self.power_transformer = power_transformer
                else:
                    self.power_transformer = PowerNormalTransformer(missing_value_threshold=self.missing_value_threshold,
                                                                    col_time=self.col_time,
                                                                    col_name="stream_watts",
                                                                    method=self.method_power,
                                                                    max_ppr=self.max_ppr,
                                                                    rolling=self.roll,
                                                                    norm_data = self.norm_data,
                                                                    id_col = self.id_col,
                                                                    keep_cols = self.keep_cols
                                                                    )
            elif i == 'stream_speed':
                self.fields.append(FieldType.SPEED)
                print('Speed not implemented yet')

        

        

    def fit(self, X):
        self.X_ = X
        for field in self.fields:
            if field == FieldType.HR:
                self.hr_transformer.fit(X)
            elif field == FieldType.POWER:
                self.power_transformer.fit(X)
        return self

    def transform(self, X):
        for field in self.fields:
            if field == FieldType.HR:
                transfo_hr = self.hr_transformer.transform(X)
            elif field == FieldType.POWER:
                transfo_power = self.power_transformer.transform(X)

        if len(self.fields) == 1:
            return transfo_hr if field == FieldType.HR else transfo_power
        else:
            # print(transfo_hr.head(), transfo_power.head())
            return transfo_hr.merge(transfo_power, on=[self.id_col, self.col_time])
    


class PowerOutliersTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_limit=3, ref_ppr=30, col_name='stream_watts'):
        self.n_limit = n_limit
        self.ref_ppr = ref_ppr
        self.col_name = col_name
        self.max_ppr = None
        self.X_ = None

    def fit(self, X, y=None):
        self.X_ = X.groupby('id')[self.col_name].agg(lambda x: x.rolling(self.ref_ppr).mean().max())
        self.max_ppr = self.X_.mean() + self.n_limit*self.X_.std()
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        ppr = X.groupby('id')[self.col_name].agg(lambda x: x.rolling(self.ref_ppr).mean().max)
        id2drop = ppr[ppr > self.max_ppr].index.to_list()
        return X[~X['id'].isin(id2drop)]


class AnormalPointsTransformer(TransformerMixin):
    def __init__(self, fields, states_to_delete=[1, 2, 3], state_field_name='states_4componnents'):
        self.fields = fields
        self.states_to_delete = states_to_delete
        self.state_field_name=state_field_name
        self.X_ = None
        self.means = pd.DataFrame()
        self.stds = pd.DataFrame()

    def fit(self, X: pd.DataFrame, y=None):
        self.X_ = X
        self.means = X.groupby(self.state_field_name)[self.fields].mean()
        self.stds = X.groupby(self.state_field_name)[self.fields].std()
        return self

    def transform(self, X:pd.DataFrame, y=None):
        for this_f in self.fields:
            X[this_f+'_transformed'] = X.apply(lambda row: (row[this_f]
                                                            - self.means.loc[row[self.state_field_name], this_f])
                                                           / self.stds.loc[row[self.state_field_name], this_f]
                                               , axis=1)
        return X[[this_f+'_transformed' for this_f in self.fields]]



def delete_nans(X, missing_value_threshold, col_name, id_col='id'):
    nan_hr_trainings = X.groupby(id_col).agg(lambda x: x.isnull().sum())[col_name]
    total_length_training = X.groupby(id_col).agg(len)[col_name]

    id_to_delete = nan_hr_trainings.loc[
        nan_hr_trainings / total_length_training > missing_value_threshold].index.values
    return X[~X[id_col].isin(id_to_delete)]
