import pandas as pd
import numpy as np
from scipy.optimize import minimize

column_names = ['variance', 'skewness', 'curtosis', 'entropy', 'label']

train_data = pd.read_csv('ML HW 4/bank-note/train.csv', header=None, names=column_names)
test_data = pd.read_csv('ML HW 4/bank-note/test.csv', header=None, names=column_names)

X_train = train_data[['variance', 'skewness', 'curtosis', 'entropy']].values
y_train = train_data['label'].map({1: 1, 0: -1}).values
X_test = test_data[['variance', 'skewness', 'curtosis', 'entropy']].values
y_test = test_data['label'].map({1: 1, 0: -1}).values

def gaussian_kernel(x1, x2, gamma):
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

def compute_kernel_matrix(X, gamma):
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = gaussian_kernel(X[i], X[j], gamma)
    return K

def svm_dual_gaussian(X, y, C, gamma):
    K = compute_kernel_matrix(X, gamma)
    
    def objective(alpha):
        return 0.5 * np.sum((alpha * y)[:, None] * (alpha * y)[None, :] * K) - np.sum(alpha)
    
    constraints = {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y)}
    bounds = [(0, C) for _ in range(len(y))]
    
    result = minimize(objective, np.zeros(len(y)), bounds=bounds, constraints=constraints)
    
    alphas = result.x
    sv_indices = alphas > 1e-5
    b = np.mean(y[sv_indices] - np.sum((alphas * y)[:, None] * K[:, sv_indices], axis=0))
    
    return alphas, b, sv_indices

def predict_svm(X_train, y_train, X_test, alphas, b, gamma):
    predictions = []
    for x_test in X_test:
        decision = sum(alphas[i] * y_train[i] * gaussian_kernel(X_train[i], x_test, gamma) for i in range(len(alphas))) + b
        predictions.append(np.sign(decision))
    return np.array(predictions)

gamma_values = [0.1, 0.5, 1, 5, 100]
C_values = [100 / len(X_train), 500 / len(X_train), 700 / len(X_train)]

best_test_error = float('inf')
best_params = None

for gamma in gamma_values:
    for C in C_values:
        alphas, b, sv_indices = svm_dual_gaussian(X_train, y_train, C, gamma)
        train_pred = predict_svm(X_train, y_train, X_train, alphas, b, gamma)
        test_pred = predict_svm(X_train, y_train, X_test, alphas, b, gamma)
        
        train_error = np.mean(train_pred != y_train)
        test_error = np.mean(test_pred != y_test)
        
        print(f"Gamma: {gamma}, C: {C}")
        print(f"Training Error: {train_error:.5f}")
        print(f"Test Error: {test_error:.5f}")
        print("------------------------------------------------------")
        
        if test_error < best_test_error:
            best_test_error = test_error
            best_params = {'gamma': gamma, 'C': C}

print(f"\nBest Parameters: {best_params}, Best Test Error: {best_test_error:.5f}")
