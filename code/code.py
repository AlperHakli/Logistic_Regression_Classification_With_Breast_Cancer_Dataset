import numpy as np

import libraries as lb



#load data
df = lb.pd.read_csv("C:/Users/Alper/Documents/GitHub/LogisticRegressionWithBreastCancerDataset/dataset/breast-cancer.csv")

#info
#df.info()

# change diagnosis contents (M = 1 , B = 0) , and delete id column
df.diagnosis = [1  if each == "M" else 0 for each in df.diagnosis]
df.drop(["id"],axis = 1,inplace = True)

#split data
y = df.diagnosis
x = df.drop(["diagnosis"],axis=1)
x_train , x_test , y_train , y_test = lb.train_test_split(x,y,test_size=0.2 , random_state= 42)
x_train = x_train.T
x_test = x_test.T

print(x_train.shape)
def sigmoid(z):
    y_head = 1/(1-lb.np.exp(-z))
    return y_head
def init_weight_and_bias(dimension):
    w = lb.np.full((dimension,1) , 0.01)
    b = 0.0
    return w,b

def forward_and_backward_propagation(w,b,x_train,y_train):
    # for forward propagation
    z = np.dot(w.T , x_train)+b
    y_head = sigmoid(z)
    loss = - ((1-y_train)*np.log(1-y_head)) - (y_train*np.log(y_head))
    cost = np.sum(loss)/x_train.shape[1]


