import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from operator import add
from pandas.io.formats import style
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

forest = RandomForestClassifier(n_estimators=1000, n_jobs=-1, class_weight='balanced')

feat_selector = BorutaPy(forest, n_estimators='auto', verbose=2)

feat_selector.fix(X,y)