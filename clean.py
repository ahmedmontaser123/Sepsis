
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
import numpy as np


class Cleaner:
    import pandas as pd
    import numpy as np
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.mixture import GaussianMixture
    def __init__(self, threshold):
        self.threshold = threshold

    def selectcol(self, X_train, y_train=None) -> list:
        X_train = X_train.copy()
        n_rows = X_train.shape[0]
        self.columns_to_drop = [
            col for col in X_train.columns if (X_train[col].isna().sum() / n_rows) * 100 >= self.threshold
        ]

        return self.columns_to_drop


       
    def guassian_fit(self,x_train):
        self.imputer = SimpleImputer(strategy='mean')
        x_train_imputed = self.imputer.fit_transform(x_train)
        self.gmm = GaussianMixture(n_components=2, covariance_type='full',random_state=42)
        self.gmm.fit(x_train_imputed)
        return self.gmm
        
    def gaussian_impute(self, x):
        if self.gmm is None:
            raise ValueError("GMM model not fitted. Call gaussian_fit first.")

        x_imputed = self.imputer.transform(x)  # initial median imputation
        resp = self.gmm.predict_proba(x_imputed)

        # fill missing using weighted mean
        for i in range(x_imputed.shape[0]):
            missing_idx = np.isnan(x.iloc[i])
            for j in np.where(missing_idx)[0]:
                weighted_mean = sum(resp[i, k] * self.gmm.means_[k, j]
                                    for k in range(self.gmm.n_components))
                x_imputed[i, j] = weighted_mean

        return pd.DataFrame(x_imputed, columns=x.columns, index=x.index)





    def impute_median(self, X_train, y_train) -> dict:
        X_train = X_train.copy()
        X_train['SepsisLabel'] = y_train

        Infants = (X_train['age_in_year'] <= 2)
        children = (X_train['age_in_year'] > 2) & (X_train['age_in_year'] <= 9)
        before_teen = (X_train['age_in_year'] >= 10) & (X_train['age_in_year'] <= 14)
        teen = (X_train['age_in_year'] >= 15) & (X_train['age_in_year'] <= 21)
        age_group = {'Infants': Infants, 'children': children, 'before_teen': before_teen, 'teen': teen}

        exclude = ['visit_occurrence_id','person_id','visit_start_date','birth_datetime',
                   'age_in_months','gender','SepsisLabel','age_in_year']
        self.dic = {}

        for col in X_train.columns:
            if col in exclude:
                continue
            self.dic[col] = {}
            for age_name, group_cond in age_group.items():
                    for gender in X_train['gender'].unique():
                        key = f"{age_name}_{gender}"
                        median_val = X_train.loc[group_cond & (X_train['gender']==gender), col].median()
                        self.dic[col][key] = median_val

        return self.dic 

    def transform_impute_median(self, df: pd.DataFrame,dic_col) -> pd.DataFrame:
        df = df.copy()
        Infants = (df['age_in_year'] <= 2)
        children = (df['age_in_year'] > 2) & (df['age_in_year'] <= 9)
        before_teen = (df['age_in_year'] >= 10) & (df['age_in_year'] <= 14)
        teen = (df['age_in_year'] >= 15) & (df['age_in_year'] <= 21)

        age_group = {
            'Infants': Infants,
            'children': children,
            'before_teen': before_teen,
            'teen': teen
        }

        for col, col_mapping in dic_col.items():
            if col not in df.columns:
                continue
            for age_name, group_condition in age_group.items():
                for gender in df['gender'].unique():
                    key = f"{age_name}_{gender}"
                    median_val = col_mapping.get(key, None)
                    if median_val is not None:
                        condition = group_condition & (df['gender'] == gender)
                        df.loc[condition, col] = df.loc[condition, col].fillna(median_val)

        return df