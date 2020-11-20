import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.linear_model import LinearRegression, RANSACRegressor

# to display full row of data
pd.set_option("display.max_column", 14)

header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.read_csv("data/housing.csv", header=None, delimiter=r"\s+", names=header)
# print(df.head())
# print(df.describe())

# EXPLORATORY DATA ANALYSIS  (EDA)

# sb.pairplot(df, height=1.5)
# study_set = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM']
# study_set = ['PTRATIO', 'B', 'LSTAT', 'MEDV']
# sb.pairplot(df[study_set], height=2.5)
# plt.show()

# CORRELATION AND FEATURE SELECTION
pd.options.display.float_format = '{:,.3f}'.format
# print(df.corr())  # using pandas dataframe
# study_set = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'MEDV']
# plt.figure()
# sb.heatmap(df[study_set].corr(), annot=True, cmap='OrRd_r')
# plt.show()

# LINEAR REGRESSION
# print(df['RM'].values)
X = df['RM'].values.reshape(-1, 1)
# print(X)
y = df['MEDV'].values

model = LinearRegression()

model.fit(X, y)
print("linear regression Coefficient:", model.coef_)
print("linear regression Intercept:", model.intercept_)

# plt.figure()
# plt.suptitle("RM v/s MEDV plot")
# sb.regplot(x=X, y=y)
# plt.xlabel("average number of rooms per dwelling.")
# plt.ylabel("median value of owner-occupied homes in \$1000s.")
# plt.show()
# plt.close()

# print("Prediction:", model.predict(np.array([9]).reshape(1, -1)))

# sb.jointplot(x='RM', y='MEDV', data=df, kind='reg')
# plt.show()
# plt.close()

# Robust Regression
# RANSAC(RANdom SAmple Consensus Algorithm)
ransac = RANSACRegressor()
ransac.fit(X, y)

# Find outliers(just for plotting)
inlier = ransac.inlier_mask_
outlier = np.logical_not(inlier)

# prediction
line_x = np.arange(3, 10, 1)  # use range of the actual data
line_y = ransac.predict(line_x.reshape(-1, 1))

# plotting
sb.set(style='darkgrid', context='notebook')
plt.scatter(X[inlier], y[inlier], c='blue', marker='o', label='inliers')
plt.scatter(X[outlier], y[outlier], c='green', marker='s', label='outliers')
plt.plot(line_x, line_y, color='red')
plt.xlabel("average number of rooms per dwelling.")
plt.ylabel("median value of owner-occupied homes in \$1000s.")
plt.legend(loc='upper left')
plt.show()

print("ransac coefficient: ", ransac.estimator_.coef_)
print("ransac intercept: ", ransac.estimator_.intercept_)
