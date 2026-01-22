from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from clean import Cleaner
import pandas as pd
import numpy as np

class preprocessorCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=90):
        self.threshold = threshold
        self.clean = Cleaner()
        self.columns_to_drop = []
        self.dic = {}
        self.processor = None
        self.gmm = None
    

    def scale_temp(self, temp):
        if 30 <= temp <= 43:
            return temp

        elif 86 <= temp <= 120:
            return (temp - 32) * 5 / 9

        elif 121 <= temp < 1000:
            return temp / 10
    
        elif temp > 1000 and temp < 10000:
            return temp / 100
    
        elif temp >= 10000:
            return temp / 1000
    
    
        else:
            return temp


    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.columns_to_drop = self.clean.selectcol(X_train, y_train)
        corr_drop = self.clean.col_corr(X_train, y_train)
        self.columns_to_drop.extend(corr_drop)
        numeric_cols = X_train.drop(columns=self.columns_to_drop).select_dtypes(include='number')
        categorical_cols = X_train.select_dtypes(include='object')
        self.gmm = self.clean.guassian_fit(numeric_cols)
        self.processor = ColumnTransformer([
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols.columns.to_list()),
            ('num', RobustScaler(), numeric_cols.columns.to_list())
        ])
        train = X_train.drop(columns=self.columns_to_drop)
        train_numeric = train.select_dtypes(include='number')
        train_categorical = train.select_dtypes(include='object')
        train_numeric = self.clean.gaussian_impute(train_numeric)
        X_train = pd.concat([train_numeric, train_categorical], axis=1)
        self.processor.fit(X_train)
        X_train['vitalBody temperaturemax'] = X_train['vitalBody temperaturemax'].apply(self.scale_temp)
        return self

    def transform(self, x: pd.DataFrame):
        x_cleaned = x.drop(columns = self.columns_to_drop)
        x_numeric = x_cleaned.select_dtypes(include='number')
        x_categorical = x_cleaned.select_dtypes(include='object')
        x_numeric_imputed = self.clean.gaussian_impute(x_numeric)
        x_cleaned = pd.concat([x_numeric_imputed, x_categorical], axis=1)
        x_cleaned['vitalBody temperaturemax'] = x_cleaned['vitalBody temperaturemax'].apply(self.scale_temp)
        x_transform = self.processor.transform(x_cleaned)


        return x_transform

