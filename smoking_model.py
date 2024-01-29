
import numpy as np
import pandas as pd
import lightgbm
import optuna
import joblib
import warnings
warnings.filterwarnings("ignore")

# Reading

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


train.drop("id",axis=1, inplace=True)
test.drop("id",axis=1, inplace=True)


# Cleaning

train.columns = train.columns.str.replace(" ", "_").str.replace(r'\(|\)', '', regex=True)
test.columns = test.columns.str.replace(" ", "_").str.replace(r'\(|\)', '', regex=True)


train.isnull().sum()

train.duplicated().sum()




num_cols = [col for col in train.columns if (train[col].dtype in ["int64","float64"]) & (train[col].nunique()>10)]
num_cols

cat_cols = [col for col in train.columns if train[col].nunique()<10]
cat_cols

# Outlier Control

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


check_outlier(train, num_cols)

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_thresholds(train, col)

check_outlier(train, num_cols)


# feature selections

def filter_correlated_variables(dataframe, target_variable, high_threshold=0.9, low_threshold=0.1):
    cor_matrix = dataframe.corr().abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    to_drop_high = [column for column in upper_triangle_matrix.columns if any(upper_triangle_matrix[column] > high_threshold)]
    cor_with_target = dataframe.corrwith(dataframe[target_variable]).abs()
    to_drop_low = cor_with_target[cor_with_target < low_threshold].index.tolist()
    dataframe_filtered = dataframe.drop(to_drop_high + to_drop_low, axis=1)

    return dataframe_filtered, to_drop_low

train,to_drop_low  = filter_correlated_variables(train, target_variable='smoking', high_threshold=0.9, low_threshold=0.1)

to_drop_low
train.shape

test.shape
test = test.drop(to_drop_low, axis=1)


# MODELING

# Split
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
X = train.drop("smoking", axis=1)
y = train["smoking"]


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1, random_state = 400)

# LightGBM

lgb = lightgbm.LGBMClassifier(metric = "auc")
lgb.fit(X_train, y_train)
roc_auc_score(y_test,lgb.predict_proba(X_test)[:,1])
# 0.860


joblib.dump(lgb, 'model.joblib')

