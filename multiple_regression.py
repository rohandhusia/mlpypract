import pandas as pd
from sklearn.datasets import load_boston
import statsmodels.api as sm
import statsmodels.formula.api as smf

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
print(mlr.summary())
