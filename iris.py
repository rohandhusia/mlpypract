import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sys

sb.set(color_codes=True)

# print(np.__version__)
# print(pd.__version__)
# print(sb.__version__)
# print(sys.version)

# reading data from local
# df = pd.read_csv("data/iris.csv")

# reading data using seaborn
iris = sb.load_dataset('iris')
print(iris.head())
print(iris.describe())
print(iris.info())
print(iris.groupby('species').size())

# visualization
sb.pairplot(iris, hue='species')
# plt.show()

iris.hist()
# plt.show()

plt.figure()
plt.subplot(2, 2, 1)
sb.violinplot(x='species', y='sepal_length', data=iris)
plt.subplot(2, 2, 2)
sb.violinplot(x='species', y='sepal_width', data=iris)
plt.subplot(2, 2, 3)
sb.violinplot(x='species', y='petal_length', data=iris)
plt.subplot(2, 2, 4)
sb.violinplot(x='species', y='petal_width', data=iris)
# plt.show()

iris.boxplot(by='species')
# plt.show()

pd.plotting.scatter_matrix(iris)
# plt.show()

sb.pairplot(iris, hue='species', diag_kind='kde')
plt.show()
