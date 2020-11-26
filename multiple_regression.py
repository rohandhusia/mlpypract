import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.datasets import load_boston

pd.set_option("display.max_column", 14)
boston_data = load_boston()
df = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)

X = df
y = boston_data.target

# METHOD 1: Statsmodels
X_constant = sm.add_constant(X)
model = sm.OLS(y, X_constant)
lr = model.fit()
# print(lr.summary())

form_lr = smf.ols(formula='y ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT',
                  data=df)
mlr = form_lr.fit()
# print(mlr.summary())

# Finding Collinearity in the data

# using correlation matrix
pd.options.display.float_format = '{:,.4f}'.format
corr_mat = df.corr()
print(corr_mat)
corr_mat[np.abs(corr_mat) < 0.6] = 0  # mask values less than 0.6 and greater than -0.6
print(corr_mat)
sb.heatmap(corr_mat, annot=True, cmap='YlGnBu')
plt.show()
plt.close()

# using eigenvector
eigenvalues, eigenvectors = np.linalg.eig(df.corr())
print(pd.Series(eigenvalues).sort_values())
print(np.abs(pd.Series(eigenvectors[:, 8])).sort_values(ascending=False))
print(df.columns[9], df.columns[8], df.columns[2], )
