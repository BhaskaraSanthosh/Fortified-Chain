import numpy as np
import pandas as pd
from sklearn import model_selection
import random
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
npx.set_np()

def main():
    filename = './dataset/cleveland.csv'
    data = pd.read_csv(filename, sep = ',', skiprows = 18)  
    data = data.to_numpy()

    k = 5
    for i in range(k):
        test_data, train_data = k_fold(data,k)
        MLP(test_data, train_data)
        
    
def k_fold(data, k):
    random.shuffle(data)
    feature = data[:, :-1]
    label = data[:, -1]
    test_data = []
    train_data = []
    for i in range(len(feature)):
        if (i % k == 0):
            test_data.append(feature[i])
        else:
            train_data.append(feature[i])   
    return test_data, train_data


def MLP(test_data, train_data):
    num_inputs, num_outputs, num_hiddens = len(train_data[0]), 10, 2
    W1 = np.random.normal(scale=0.01, size=(num_inputs, num_hiddens))
    b1 = np.zeros(num_hiddens)
    W2 = np.random.normal(scale=0.01, size=(num_hiddens, num_outputs))
    b2 = np.zeros(num_outputs)
    params = [W1, b1, W2, b2]
    for param in params:
        param.attach_grad()
    def relu(X):
        return np.maximum(X, 0)
    def net(X):
        X = X.reshape((-1, num_inputs))
        H = relu(np.dot(X, W1) + b1)
        return np.dot(H, W2) + b2
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    num_epochs, lr = 10, 0.1
    

    

main()    