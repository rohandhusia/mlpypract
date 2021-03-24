import pandas as pd
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

pd.set_option("display.max_column", 14)
boston_data = load_boston()
df = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
X = df['LSTAT'].values.reshape(-1, 1)
y = boston_data.target
# print((df.head()))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X_train, y_train)

y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

print(
    "MSE train:{}, test:{}".format(mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
print("R^2 train:{}, test:{}".format(r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))

result = pd.DataFrame(tree.feature_importances_, df.columns)
result.columns = ['features']
result.sort_values(by='features', ascending=False).plot(kind='bar')
