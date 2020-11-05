import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sys

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

