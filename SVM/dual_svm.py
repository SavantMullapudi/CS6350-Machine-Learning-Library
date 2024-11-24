import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from scipy.optimize import minimize

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
    b = 0
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

def dual_objective(alpha, X, y, C):
    n, d = X.shape
    w = np.dot(alpha * y, X)
    hinge_loss = np.maximum(1 - np.dot(X, w), 0)
    regularization_term = 0.5 * np.dot(w, w)
    
    dual_objective_value = C * np.sum(hinge_loss) + regularization_term
    return dual_objective_value

alpha_init = np.zeros(len(X_train))

for C in C_range:
    w_primal, b_primal = svm_primal(X_train, y_train, C, gamma_rate, a, max_epochs)

    train_predictions_primal = np.sign(np.dot(X_train, w_primal) + b_primal)
    train_error_primal = np.mean(train_predictions_primal != y_train)

    test_predictions_primal = np.sign(np.dot(X_test, w_primal) + b_primal)
    test_error_primal = np.mean(test_predictions_primal != y_test)

    print(f"Primal SVM Results for C = {C}")
    print(f"w_primal: {w_primal}")
    print(f"b_primal: {b_primal}")
    print(f"Training Error: {train_error_primal:.4f}")
    print(f"Test Error: {test_error_primal:.4f}")
    print('------')


    result = minimize(dual_objective, alpha_init, args=(X_train, y_train, C), bounds=[(0, C) for _ in range(len(X_train))])
    alpha_optimal = result.x

    w_dual = np.dot(alpha_optimal * y_train, X_train)
    b_dual = y_train - np.dot(X_train, w_dual)
    b_dual = np.mean(b_dual)

    train_predictions_dual = np.sign(np.dot(X_train, w_dual) + b_dual)
    train_error_dual = np.mean(train_predictions_dual != y_train)

    test_predictions_dual = np.sign(np.dot(X_test, w_dual) + b_dual)
    test_error_dual = np.mean(test_predictions_dual != y_test)

    print(f"Dual SVM Results for C = {C}")
    print(f"w_dual: {w_dual}")
    print(f"b_dual: {b_dual}")
    print(f"Training Error: {train_error_dual:.4f}")
    print(f"Test Error: {test_error_dual:.4f}")
    print('------')

    weight_difference = np.linalg.norm(w_primal - w_dual)
    bias_difference = np.abs(b_primal - b_dual)
    train_error_difference = np.abs(train_error_primal - train_error_dual)
    test_error_difference = np.abs(test_error_primal - test_error_dual)

    print("Differences between Primal and Dual SVM:")
    print(f"Feature Weights Difference: {weight_difference:.4f}")
    print(f"Bias Difference: {bias_difference:.4f}")
    print(f"Training Error Difference: {train_error_difference:.4f}")
    print(f"Test Error Difference: {test_error_difference:.4f}")
    print("---------------------------------------")
