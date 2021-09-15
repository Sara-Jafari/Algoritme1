import tensorflow
import keras
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import pickle

data = pd.read_csv("C:/Users/mmm/Desktop/New folder/student/student-mat.csv", sep=";")
print(data.head())
data = data[["G1", "G2", "G3", "Walc", "health", "absences"]]
predict = "G3"
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, y_train, x_test, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
best = 0
for _ in range(10):
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    # print(acc)
    if acc > best:
        best = acc
        with open("student model.pickle ", "wb") as f:
            pickle.dump(linear, f)
            print("best is=", best)

