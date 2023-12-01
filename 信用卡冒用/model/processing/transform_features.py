import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class TimeTransformer(BaseEstimator, TransformerMixin):
    """ Normalize time, e.g. 120000 => 0.5"""
    def __init__(self, variables: str):
        self.variables = variables

    def fit(self, X:pd.DataFrame, y:pd.Series=None):
        return self

    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        def to_second(x):
            hours = x // 10000
            minutes = (x % 10000) // 100
            seconds = x % 100
            total_seconds = hours * 3600 + minutes * 60 + seconds
            return total_seconds
            
        X = X.copy()
        X[self.variables] = X[self.variables].apply(to_second)
        scaler = MinMaxScaler()
        X[self.variables] = scaler.fit_transform(X[self.variables].values.reshape(-1, 1))
        # normalized_loctm = X['normalized_loctm']
        # X.drop([self.time, 'normalized_loctm'], axis=1, inplace=True)
        # X.insert(1, self.time, normalized_loctm)

        return X
    
class NewNAColumn(BaseEstimator, TransformerMixin):
    """ create new column for na value which indicates if it is na 1 for nan value 0 for not nan value"""
    def __init__(self, col):
        self.col = col
        
    def fit(self, X=None, y=None):
        return self
    
    def transform(self, X, y=None):
        for col in self.col:
            X[col+'_na'] = np.where(X[col].isnull(), 1, 0)
            X[col] = X[col].fillna(X[col].nunique())
        return X

        
    
        