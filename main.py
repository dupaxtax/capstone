import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from operator import add
from pandas.io.formats import style
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter


data = pd.read_csv('data/framingham.csv')
data.drop(['education'],axis=1,inplace=True)
data.head()

# X contains all the values for all columns except the TenYearCHD
X = data.iloc[:,:-1].values
# y cotains the values for column TenYearCHD
y = data.iloc[:,-1].values

print("Something to stop at")
# forest = RandomForestClassifier(n_estimators=1000, n_jobs=-1, class_weight='balanced')
# 
# feat_selector = BorutaPy(forest, n_estimators='auto', verbose=2)
# 
# feat_selector.fit(X, y)
# 
# most_important = data.columns[:1][feat_selector.support_].toList()
# 
# top_features = data.columns[:-1][feat_selector.ranking_ <=6].toList()
# 
# X = data[top_features]
# y = data.iloc[:,-1]
# 
# num_before = dict(Counter(y))
# 
# over = SMOTE(sampling_strategy=0.8)
# under = RandomUnderSampler(sampling_strategy=0.8)
# steps = [('o', over), ('u', under)]
# pipeline = Pipeline(steps=steps)
# 
# X_smote, y_smote = pipeline.fit_resample(X,y)
# 
# num_after=dict(Counter(y_smote))
# print(num_before, num_after)