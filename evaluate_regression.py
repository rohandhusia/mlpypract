import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.datasets import load_boston
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

pd.set_option("display.max_column", 14)

header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.read_csv("data/housing.csv", header=None, delimiter=r"\s+", names=header)
pd.options.display.float_format = '{:,.3f}'.format

X = df.iloc[:, :-1].values
y = df['MEDV'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

# Method 1: Residual analysis
sb.set(style='darkgrid', context='notebook')
plt.figure('Residual analysis')
plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='orange', marker='*', label='test data')
plt.xlabel('Predited Values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.show()

# Method 2: Mean Squared Error
print("Mean Squared Error")
print("training data: ", mean_squared_error(y_train, y_train_pred))
print("test data: ", mean_squared_error(y_test, y_test_pred))

# Method 3: Coefficient of Determination
print("Coefficient of Determination")
print("training data: ", r2_score(y_train, y_train_pred))
print("test data: ", r2_score(y_test, y_test_pred))
