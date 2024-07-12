import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import StandardScaler

class PowerNormalTransformer(BaseEstimator, TransformerMixin):

    def __init__(self,
                 missing_value_threshold=0.75,
                 col_name='stream_watts',
                 col_time='stream_time',
                 method='ppr30',
                 max_ppr=800,
                 ):
        self.col_name = col_name
        self.col_time = col_time
        self.missing_value_threshold = missing_value_threshold
        self.method = method
        self.ppr = 0
        self.max_ppr = max_ppr

    def fit(self, X):
        self.ppr = X.groupby('id', group_keys=False)[self.col_name].rolling(30).mean().max()
        if self.ppr > 1000:
            print('ATTENTION: PPR: {}'.format(self.ppr))
            self.ppr = self.max_ppr
        self.X_ = X
        return self

    def transform(self, X):
        check_is_fitted(self)
        tmp_X = X[['id', self.col_time, self.col_name]]
        tmp_X.loc[:, self.col_name] = tmp_X[self.col_name] / self.ppr

        # Delete the trainings with too many nans
        tmp_X = delete_nans(tmp_X, self.missing_value_threshold, self.col_name)

        # Filling and Smoothing the power
        tmp_X = tmp_X.set_index(['id', self.col_time])
        tmp_X = tmp_X.groupby('id', group_keys=False).apply(
            lambda x: x.fillna(0))
        tmp_X = tmp_X.groupby('id', group_keys=False).apply(
            lambda x: x.rolling(30).mean())
        return tmp_X.reset_index()


class HRNormalTransformer(BaseEstimator, TransformerMixin):

    def __init__(self,
                 missing_value_threshold=0.75,
                 col_name='stream_heartrate',
                 col_time='stream_time',
                 ):
        self.missing_value_threshold = missing_value_threshold
        self.col_name = col_name
        self.col_time = col_time
        self.mean = 0
        self.std = 0

    def fit(self, X):
        self.mean = X[self.col_name].mean()
        self.std = X[self.col_name].std()
        self.X_ = X
        return self

    def transform(self, X):
        check_is_fitted(self)

        tmp_X = X[['id', self.col_time, self.col_name]]
        tmp_X.loc[:, self.col_name] = (tmp_X[self.col_name] - self.mean) / (2 * self.std)

        tmp_X = delete_nans(tmp_X, self.missing_value_threshold, self.col_name)

        # Filling and Smoothing the power
        tmp_X = tmp_X.set_index(['id', self.col_time])
        tmp_X = tmp_X.groupby('id', group_keys=False).apply(
            lambda x: x.interpolate())
        return tmp_X.reset_index()


class AthleteTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, col_hr='stream_heartrate',
                 col_power='stream_watts',
                 col_time='stream_time',
                 hr_transformer=None,
                 power_transformer=None,
                 missing_value_threshold=0.75,
                 method_power='ppr30',
                 max_ppr=800,
                 ):

        self.missing_value_threshold = missing_value_threshold
        self.col_time = col_time
        self.col_hr = col_hr
        self.col_power = col_power
        self.method_power = method_power
        self.max_ppr = max_ppr
        if hr_transformer:
            self.hr_transformer = hr_transformer
        else:
            self.hr_transformer = HRNormalTransformer(missing_value_threshold=self.missing_value_threshold,
                                                      col_time=self.col_time,
                                                      col_name=self.col_hr
                                                      )

        if power_transformer:
            self.power_transformer = power_transformer
        else:
            self.power_transformer = PowerNormalTransformer(missing_value_threshold=self.missing_value_threshold,
                                                            col_time=self.col_time,
                                                            col_name=self.col_power,
                                                            method=self.method_power,
                                                            max_ppr=self.max_ppr,
                                                            )

    def fit(self, X):
        self.X_ = X
        self.hr_transformer.fit(X)
        self.power_transformer.fit(X)
        return self

    def transform(self, X):
        transfo_hr = self.hr_transformer.transform(X)
        transfo_power = self.power_transformer.transform(X)
        return transfo_hr.merge(transfo_power, on=['id', self.col_time])


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



def delete_nans(X, missing_value_threshold, col_name):
    nan_hr_trainings = X.groupby('id').agg(lambda x: x.isnull().sum())[col_name]
    total_length_training = X.groupby('id').agg(len)[col_name]

    id_to_delete = nan_hr_trainings.loc[
        nan_hr_trainings / total_length_training > missing_value_threshold].index.values
    return X[~X.id.isin(id_to_delete)]
