import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

pd.set_option("display.max_column", 14)

header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.read_csv("data/housing.csv", header=None, delimiter=r"\s+", names=header)
pd.options.display.float_format = '{:,.3f}'.format

X = df.iloc[:, :-1].values
y = df['MEDV'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("LINEAR")
lr = LinearRegression()
lr.fit(X_train, y_train)

y_train_pred_lr = lr.predict(X_train)
y_test_pred_lr = lr.predict(X_test)

# Method 1: Residual analysis
sb.set(style='darkgrid', context='notebook')
plt.figure('Residual analysis')
plt.scatter(y_train_pred_lr, y_train_pred_lr - y_train, c='blue', marker='o', label='training data')
plt.scatter(y_test_pred_lr, y_test_pred_lr - y_test, c='orange', marker='*', label='test data')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.show()
plt.close()

# Method 2: Mean Squared Error
print("Mean Squared Error")
print("training data: ", mean_squared_error(y_train, y_train_pred_lr))
print("test data: ", mean_squared_error(y_test, y_test_pred_lr))

# Method 3: Coefficient of Determination
print("Coefficient of Determination")
print("training data: ", r2_score(y_train, y_train_pred_lr))
print("test data: ", r2_score(y_test, y_test_pred_lr))
