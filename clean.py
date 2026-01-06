
class Cleaner:
    import pandas as pd
    def __init__(self, threshold=90):
        self.threshold = threshold

    def selectcol(self, X_train, y_train=None) -> list:
        X_train = X_train.copy()
        X_train['SepsisLabel'] = y_train
        n_rows = X_train.shape[0]
        self.columns_to_drop = [
            col for col in X_train.columns if (X_train[col].isna().sum() / n_rows) * 100 >= self.threshold
        ]

        return self.columns_to_drop


    def col_corr(self, X_train, y_train) -> list:
        X_train = X_train.copy()
        X_train['SepsisLabel'] = y_train
        numeric_df = X_train.select_dtypes(include='number')
        n_rows = X_train.shape[0]
        self.cols_drop_corr = [
            col for col in numeric_df.columns
               if (abs(numeric_df[col].corr(numeric_df['SepsisLabel'])) < 0.09 and
                  (numeric_df[col].isna().sum() / n_rows) * 100 >= 85)
        ]
        return self.cols_drop_corr

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