from sklearn.preprocessing import StandardScaler, RobustScaler,OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer


class SCaler(BaseEstimator,TransformerMixin):
    '''
    Description:
    This class is to apply scaling and encoding to the data in a pipeline of processing and modeling.
    Methods:
    - scale_temp: A helper function to convert temperature values to a consistent scale (Celsius) based on specified rules.
    - fit: fits the scaler and encoder to the training data, identifying numeric and categorical columns and preparing the transformations.
    - transform: applies the scaling and encoding transformations to the input data and returns the processed array
    
    '''
    def __init__(self):
        self.processor = None


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
    
    def fit(self,x_train,y_train = None):

        if 'vitalBody temperaturemax' in x_train.columns:
            x_train['vitalBody temperaturemax'] = x_train['vitalBody temperaturemax'].apply(self.scale_temp)
        
        numeric = x_train.select_dtypes(include = 'number').columns.to_list()
        cat = x_train.select_dtypes(include = 'object').columns.to_list()


        self.processor = ColumnTransformer([
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat),
            ('num', StandardScaler(), numeric)
        ])
        
        

        self.processor.fit(x_train)

    
        return self
    

    def transform(self , x):
        if 'vitalBody temperaturemax' in x.columns:
            x['vitalBody temperaturemax'] = x['vitalBody temperaturemax'].apply(self.scale_temp)

        x = self.processor.transform(x)
        return x
        

        

