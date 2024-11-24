import pandas as pd
import numpy as np
from sklearn.utils import shuffle

column_names = ['variance', 'skewness', 'curtosis', 'entropy', 'label']

train_data = pd.read_csv('bank-note/train.csv', header=None, names=column_names)
test_data = pd.read_csv('bank-note/test.csv', header=None, names=column_names)

X_train = train_data[['variance', 'skewness', 'curtosis', 'entropy']].values  
y_train = train_data['label'].map({1: 1, 0: -1}).values  
X_test = test_data[['variance', 'skewness', 'curtosis', 'entropy']].values    
y_test = test_data['label'].map({1: 1, 0: -1}).values  

max_epochs = 100
C_range = [100 / len(X_train), 500 / len(X_train), 700 / len(X_train)]  
gamma_rate = 0.5
a = 0.005

def svm_primal(X, y, C, gamma_rate, a, max_epochs):

    n, d = X.shape  
    w = np.zeros(d)  
    b = 0.1
    updates = 0  

    for epoch in range(max_epochs):
        dynamic_C = C * (1 + epoch / max_epochs)
        X, y = shuffle(X, y, random_state=epoch)

        for i in range(n):
            updates += 1
            eta = gamma_rate / (1 + updates ** 0.5) 
            margin = y[i] * (np.dot(X[i], w) + b)
            
            if margin < 1:
                w = (1 - eta) * w + eta * dynamic_C * y[i] * X[i]
                b = b * (1 - eta * 0.01) + eta * dynamic_C * y[i]
            else:
                w = (1 - eta * 0.01) * w  
                
    return w, b

for i in C_range:
    w, b = svm_primal(X_train, y_train, i, gamma_rate, a, max_epochs)

    train_predictions = np.sign(np.dot(X_train, w) + b)
    train_error = np.mean(train_predictions != y_train)

    test_predictions = np.sign(np.dot(X_test, w) + b)
    test_error = np.mean(test_predictions != y_test)

    print(f"Hyperparameter C: {i}")
    print(f"Training Error: {train_error:.4f}")
    print(f"Test Error: {test_error:.4f}")
    
