import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

# creating random dataset
np.random.seed(42)
n_samples = 100

X = np.linspace(0, 10, 100)
rng = np.random.randn(n_samples) * 100

y = X ** 3 + 100 + rng  # polynomial regression (y=x**3 + 100 + error)

plt.scatter(X, y)
plt.show()
plt.close()

# modelling using linear regression
lr = LinearRegression()
lr.fit(X.reshape(-1, 1), y)
model_pred = lr.predict(X.reshape(-1, 1))

plt.scatter(X, y)
plt.plot(X, model_pred)
plt.show()
plt.close()
print("LR1 r2 score: ", r2_score(y, model_pred))

# modelling using polynomial regression
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X.reshape(-1, 1))
lr2 = LinearRegression()
lr2.fit(X_poly, y.reshape(-1, 1))
model2_pred = lr2.predict(X_poly)

plt.scatter(X, y)
plt.plot(X, model2_pred)
plt.show()
plt.close()
print("LR2 r2 score: ", r2_score(y, model2_pred))
