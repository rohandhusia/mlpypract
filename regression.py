import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.datasets import load_boston
import seaborn as sb
import matplotlib.pyplot as plt

# to display full row of data
pd.set_option("display.max_column", 14)
header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.read_csv("data/housing.csv", header=None, delimiter=r"\s+", names=header)
# print(df.head())
# print(df.describe())

# EXPLORATORY DATA ANALYSIS  (EDA)

# sb.pairplot(df, height=1.5)
# study_set = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM']
study_set = ['PTRATIO', 'B', 'LSTAT', 'MEDV']
sb.pairplot(df[study_set], height=2.5)
plt.show()
