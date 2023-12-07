import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

from typing import List

class TimeTransformer(BaseEstimator, TransformerMixin):
    """ 
    Turn hr-min-sec format to sec and then normalize with MinMaxScaler, e.g. 120000 => 0.5
    
    Parameters:
    - variables (str): the to-transform feature column name

    Methods:
    - fit(X, y=None): Fit the transformer.
    - transform(X): Turn hr-min-sec format to sec and then normalize with MinMaxScaler
    """
    def __init__(self, variables: str):
        self.variables = variables

    def fit(self, X:pd.DataFrame, y:pd.Series=None):
        """
        Fit the transformer

        Parameters:
        - X (pd.DataFrame): input DataFrame
        - y (pd.Series): input target (can be ignored)

        Returns:
        TimeTransformer: The fitted TimeTransformer instance
        """
        return self

    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """ 
        Turn hr-min-sec format to sec and then normalize with MinMaxScaler, e.g. 120000 => 0.5
        
        Parameters:
        - X (pd.DataFrame): the input DataFrame

        Returns:
        - pd.DataFrame: the transformed DataFrame
        """
        def to_second(x: int):
            """
            turn hr-min-sec into sec

            Parameters:
            - X (int): from 000000(in int format: 0) 到 235959 in the hr-min-sec format

            Returns:
            int: the transformed seconds
            """
            hours = x // 10000
            minutes = (x % 10000) // 100
            seconds = x % 100
            total_seconds = hours * 3600 + minutes * 60 + seconds
            return total_seconds
            
        X = X.copy()
        X[self.variables] = X[self.variables].apply(to_second)
        scaler = MinMaxScaler()
        X[self.variables] = scaler.fit_transform(X[self.variables].values.reshape(-1, 1))

        return X
    
class NewNAColumn(BaseEstimator, TransformerMixin):
    """ 
    A scikit-learn compatible transformer for handling missing values in a specific column.

    Parameters:
    - col (list of str): The name(s) of the column(s) to transform.

    Methods:
    - fit(X, y=None): Fit the transformer.
    - transform(X): Fill NaN values and create a new binary column indicating NaN presence.
    """
    def __init__(self, col: List[str]):
        self.col = col
        
    def fit(self, X, y=None):
        """
        Fit the transformer

        Parameters:
        - X (pd.DataFrame): input DataFrame
        - y (pd.Series): input target (can be ignored)

        Returns:
        TimeTransformer: The fitted NewNAColumn instance
        """
        return self
    
    def transform(self, X, y=None):
        """
        Transform the input DataFrame by filling NaN values and creating a new binary column indicating NaN presence.

        Parameters:
        - X (pd.DataFrame): Input DataFrame.
        - y (pd.Series): Input target (can be ignored).

        Returns:
        pd.DataFrame: Transformed DataFrame.
        """
        for col in self.col:
            X[col+'_na'] = np.where(X[col].isnull(), 1, 0)
            X[col].fillna(X[col].nunique(), inplace=True)
        return X

        
    
        