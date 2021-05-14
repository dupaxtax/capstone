import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from operator import add
from pandas.io.formats import style
import seaborn as sns

data = pd.read_csv('data/framingham.csv')
data.drop(['education'],axis=1,inplace=True)
data.head()

missing_data = data.isnull().sum()
total_percentage = (missing_data.sum()/data.shape[0]) *100
print(f'The total percentage of missing data is {round(total_percentage,2)}%')

total = data.isnull().sum().sort_values(ascending=False)
percent_total = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)*100
missing = pd.concat([total, percent_total], axis=1, keys=["Total", "Percentage"])
missing_data = missing[missing['Total']>0]
