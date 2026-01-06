from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder,RobustScaler
from sklearn.compose import ColumnTransformer
from clean import Cleaner
class preprocessorCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=90):
        self.threshold = threshold
        self.clean = Cleaner()
        self.columns_to_drop = []
        self.dic = {}
        self.prosessor = None

    def fit(self , x_train , y_train):
        self.columns_to_drop = self.clean.selectcol(x_train , y_train)
        lst1 = self.clean.col_corr(x_train , y_train)
        self.columns_to_drop.extend(lst1)
        self.dic = self.clean.impute_median(x_train , y_train)
        return self


    def transform(self , x_train):
        x = x_train.copy()
        x = x.drop(columns = self.columns_to_drop)
        x = self.clean.transform_impute_median(x , self.dic)
        numerical_col = x.select_dtypes(include = 'number').columns
        categorical_col = x.select_dtypes(exclude = 'number').columns
        self.prosessor = ColumnTransformer([('cat', OneHotEncoder(drop = 'first', handle_unknown = 'ignore') , categorical_col),
                                     ('num', RobustScaler() , numerical_col)])
        self.prosessor.fit(x_train)
        x = self.prosessor.transform(x)
        return x




