from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from clean import Cleaner
import pandas as pd
import numpy as np

class Cleaning(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.clean = Cleaner(threshold=50)
        self.columns_to_drop = []
        self.dic = {}
        self.processor = None
        self.gmm = None
        
    
    
    
    
    def fit(self, x_train: pd.DataFrame, y_train: pd.Series):
        self.columns_to_drop = self.clean.selectcol(x_train, y_train)
        x_train = x_train.drop(columns=self.columns_to_drop)
        numeric_cols = x_train.select_dtypes(include = 'number')   
        self.gmm = self.clean.guassian_fit(numeric_cols)
        
        return self
    
    def transform(self, x: pd.DataFrame):
        x = x.drop(columns=self.columns_to_drop)
        x_numeric = x.select_dtypes(include='number')
        x_categorical = x.select_dtypes(include='object')
        
        # Impute numeric features
        x_numeric_imputed = self.clean.gaussian_impute(x_numeric)
        x_cleaned = pd.concat([x_numeric_imputed, x_categorical], axis=1)
                
        return x_cleaned