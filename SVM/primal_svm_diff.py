import pandas as pd
import numpy as np
from sklearn.utils import shuffle

column_names = ['variance', 'skewness', 'curtosis', 'entropy', 'label']

train_data = pd.read_csv('ML HW 4/bank-note/train.csv', header=None, names=column_names)
test_data = pd.read_csv('ML HW 4/bank-note/test.csv', header=None, names=column_names)

X_train = train_data[['variance', 'skewness', 'curtosis', 'entropy']].values  
y_train = train_data['label'].map({1: 1, 0: -1}).values  
X_test = test_data[['variance', 'skewness', 'curtosis', 'entropy']].values    
y_test = test_data['label'].map({1: 1, 0: -1}).values  

max_epochs = 100
C_range = [100 / len(X_train), 500 / len(X_train), 700 / len(X_train)]  
gamma_rate = 0.5

def primal_optimal_svm(X, y, C, gamma_rate, max_epochs):
    n, d = X.shape  
    w = np.zeros(d)  
    b = 0
    updates = 0  

    for epoch in range(max_epochs):
        dynamic_C = C * (1 + epoch / max_epochs)  
        X, y = shuffle(X, y, random_state=epoch) 

        for i in range(n):
            updates += 1
            eta = gamma_rate / (1 + np.sqrt(updates))  
            margin = y[i] * (np.dot(X[i], w) + b)

            if margin < 1:
                w = (1 - eta) * w + eta * dynamic_C * y[i] * X[i]
                b = b * (1 - eta * 0.01) + eta * dynamic_C * y[i]
            else:
                w = (1 - eta * 0.01) * w  
                
    return w, b

def primal_optimal_svm2(X, y, C, gamma_rate, max_epochs):
    n, d = X.shape  
    w = np.zeros(d)  
    b = 0
    updates = 0  

    for epoch in range(max_epochs):
        dynamic_C = C * (1 + epoch / max_epochs) 
        X, y = shuffle(X, y, random_state=epoch)

        for i in range(n):
            updates += 1
            eta = gamma_rate / (1 + updates) 
            margin = y[i] * (np.dot(X[i], w) + b)

            if margin < 1:
                w = (1 - eta) * w + eta * dynamic_C * y[i] * X[i]
                b = b * (1 - eta * 0.01) + eta * dynamic_C * y[i]
            else:
                w = (1 - eta * 0.01) * w  

    return w, b

for i in C_range:
    w1, b1 = primal_optimal_svm(X_train, y_train, i, gamma_rate, max_epochs)
    w2, b2 = primal_optimal_svm2(X_train, y_train, i, gamma_rate, max_epochs)

    train_predictions1 = np.sign(np.dot(X_train, w1) + b1)
    train_error1 = np.mean(train_predictions1 != y_train)
    test_predictions1 = np.sign(np.dot(X_test, w1) + b1)
    test_error1 = np.mean(test_predictions1 != y_test)

    train_predictions2 = np.sign(np.dot(X_train, w2) + b2)
    train_error2 = np.mean(train_predictions2 != y_train)
    test_predictions2 = np.sign(np.dot(X_test, w2) + b2)
    test_error2 = np.mean(test_predictions2 != y_test)

    training_error_diff = abs(train_error1 - train_error2)
    test_error_diff = abs(test_error1 - test_error2)
    weight_diff = np.linalg.norm(w1 - w2)
    bias_diff = abs(b1 - b2)

    print(f"C: {i}")
    print(f"Training Error Difference: {training_error_diff:.6f}")
    print(f"Test Error Difference: {test_error_diff:.6f}")
    print(f"Weight Difference: {weight_diff:.6f}")
    print(f"Bias Difference: {bias_diff:.6f}")
    print('------')


