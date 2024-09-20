import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#load data
df = pd.read_csv("C:/Users/Alper/Documents/GitHub/LogisticRegressionWithBreastCancerDataset/dataset/breast-cancer.csv")

# preparing data
df.diagnosis = [1 if each == "M" else 0 for each in df.diagnosis]
df.drop(["id"], axis=1, inplace=True)
for x in df.columns:
    df[x] = (df[x] - np.min(df[x])) / (np.max(df[x]) - np.min(df[x]))
#split data
y = df.diagnosis.values
x = df.drop(["diagnosis"], axis=1)
for column in x.columns:
    x[column] = (x[column] - np.min(x[column])) / ((np.max(x[column]) - np.min(x[column])))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train = x_train.T
x_test = x_test.T

#sigmoid function
def sigmoid(z):
    y_head = 1 / (1 + np.exp(-z))
    return y_head


def init_weight_and_bias(dimension):
    w = np.full((dimension, 1), 0.01)
    b = 0.0
    return w, b


def forward_and_backward_propagation(w, b, x_train, y_train):
    #forward propagation
    z = (np.dot(w.T, x_train)) + b
    y_head = sigmoid(z)
    loss = - (1 - y_train) * np.log(1 - y_head) - (y_train * np.log(y_head))

    cost = np.sum(loss) / x_train.shape[1]
    #backward propagation
    derivative_weight = (np.dot(x_train, (y_head - y_train).T)) / x_train.shape[1]
    derivative_bias = (np.sum(y_head - y_train)) / x_train.shape[1]
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    return gradients, cost


def update_parameters(w, b, x_train, y_train, number_of_iterarions, learning_rate):
    cost_list = []
    cost_list2 = []
    index_list = []
    for i in range(0, number_of_iterarions, 1):
        gradients, cost = forward_and_backward_propagation(w, b, x_train, y_train)
        w = w - (learning_rate * gradients["derivative_weight"])
        b = b - (learning_rate * gradients["derivative_bias"])
        if (i % 10 == 0):
            print("cost after {} iterations : {}".format(i, cost))
    #updating parameters
    parameters = {"weight": w, "bias": b}
    return parameters


def prediction(w, b, x_test):
    z = sigmoid(np.dot(w.T, x_test) + b)
    y_prediction = np.zeros((1, z.shape[1]))
    print(y_prediction.shape)
    for i in range(z.shape[1]):

        if (z[0, i] > 0.5):
            y_prediction[0, i] = 1
        else:
            y_prediction[0, i] = 0
    return y_prediction


def logistic_regression(x_train, y_train, x_test, y_test, number_of_iterarions, learning_rate):
    dimension = x_train.shape[0]
    w, b = init_weight_and_bias(dimension)
    parameters = update_parameters(w, b, x_train, y_train, number_of_iterarions, learning_rate)
    y_prediction = prediction(parameters["weight"], parameters["bias"], x_test)
    print("test accuary rate = ", 100 - (np.mean(np.abs(y_prediction - y_test)) * 100))

#logistic_regression(x_train, y_train, x_test, y_test, 600, 0.01)

#shortcut
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=42)
lr.fit(x_train.T,y_train.T)
print(lr.score(x_train.T,y_train.T))