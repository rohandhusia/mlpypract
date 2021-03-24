import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

dataset = pd.read_csv("data/data.csv")
# print(np.where(dataset.isnull()))
# print(dataset.dtypes.value_counts())
# print(dataset.dtypes == 'object')
# print(dataset.tail())
target = dataset.diagnosis
# print(target.shape)
# print(target.head())
unwanted_list = ["id", "diagnosis"]
refined_data = dataset.drop(unwanted_list, axis=1)
# print(refined_data.head())
# print(refined_data.shape)
# ax = sns.countplot(target, label="count")
B, M = target.value_counts()
# print("number of begnin", B)
# print("number of malignant", M)
# plt.show()
# plt.close()

# # looking at correlation between target and features
# data_target = target
# # normalize data for plotting
# data_n_2 = (refined_data - refined_data.mean()) / (refined_data.std())
# # 1st 10 features
# sub_data = pd.concat([target, data_n_2.iloc[:, 0:10]], axis=1)
# # flattening data
# sub_data = pd.melt(sub_data, id_vars="diagnosis", var_name="features", value_name="value")
# sns.swarmplot(x="features", y="value", hue="diagnosis", data=sub_data)
# plt.show()
# plt.xticks(rotation=90)
# plt.close()
#
#
# # 2nd 10 features
# sub_data = pd.concat([target, data_n_2.iloc[:, 10:20]], axis=1)
# # flattening data
# sub_data = pd.melt(sub_data, id_vars="diagnosis", var_name="features", value_name="value")
# sns.swarmplot(x="features", y="value", hue="diagnosis", data=sub_data)
# plt.show()
# plt.xticks(rotation=90)
# plt.close()
#
#
# # 3rd 10 features
# sub_data = pd.concat([target, data_n_2.iloc[:, 20:30]], axis=1)
# # flattening data
# sub_data = pd.melt(sub_data, id_vars="diagnosis", var_name="features", value_name="value")
# sns.swarmplot(x="features", y="value", hue="diagnosis", data=sub_data)
# plt.show()
# plt.xticks(rotation=90)
# plt.close()


# data preprocessing
target = np.where(target.values == 'M', 0, 1)
# print(type(target))

# data normalizing
scaler = MinMaxScaler()
data = scaler.fit_transform(refined_data.values)
# print(data)

# split in train and test
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.1)

# creating algorithm
knn = KNeighborsClassifier()

# grid search
grid_pram = {
    'n_neighbors': [5, 7, 11, 15, 3],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [10, 30, 50, 100]
}

grid_search = GridSearchCV(knn, param_grid=grid_pram)
grid_search.fit(x_train, y_train)
# print(grid_search.best_estimator_)
knn = KNeighborsClassifier(n_neighbors=3, p=2, metric="minkowski", leaf_size=10)
knn.fit(x_train, y_train)

predict = knn.predict(x_test)
print("accuracy of the algorithm is: {}".format(accuracy_score(y_test, predict)))
