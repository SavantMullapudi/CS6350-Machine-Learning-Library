import numpy as np
import pandas as pd
from scipy.optimize import minimize

column_names = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
train_data = pd.read_csv('bank-note/train.csv', header=None, names=column_names)

X_train = train_data[['variance', 'skewness', 'curtosis', 'entropy']].values  
y_train = train_data['label'].map({1: 1, 0: -1}).values  

def gaussian_kernel_matrix(X, gamma):
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = np.exp(-gamma * np.sum((X[i, :] - X[j, :]) ** 2))
    return K

def train_dual_svm_gaussian(X, y, C, gamma):
    K = gaussian_kernel_matrix(X, gamma)
    
    def objective(alpha):
        return 0.5 * np.dot(alpha, np.dot(K, alpha * y) * y) - np.sum(alpha)

    constraints = {'type': 'eq', 'fun': lambda alpha: np.sum(alpha * y)}
    bounds = [(0, C) for _ in range(X.shape[0])]

    result = minimize(fun=objective,
                      x0=np.zeros(X.shape[0]),
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints)
    
    alphas = result.x
    sv = alphas > 1e-5
    b = np.mean(y[sv] - np.dot(K[sv], alphas * y))
    
    return alphas, b, sv

def count_support_vectors(support_vectors):
    return np.sum(support_vectors)

def calculate_overlap(support_vectors_1, support_vectors_2):
    return np.sum(np.logical_and(support_vectors_1, support_vectors_2))

gamma_values = [0.01, 0.1, 0.5, 1, 5, 100]
C_values = [100 / len(X_train), 500 / len(X_train), 700 / len(X_train)]

for C in C_values:
    print(f"\nC = {C:.5f}")
    support_vectors_list = []
    
    for gamma in gamma_values:
        alphas, b, sv = train_dual_svm_gaussian(X_train, y_train, C, gamma)
        num_support_vectors = count_support_vectors(sv)
        support_vectors_list.append(sv)
        print(f"Gamma: {gamma:.5f}, Number of Support Vectors: {num_support_vectors}")
    
    if C == 500 / len(X_train):
        print("\nOverlap of Support Vectors for Consecutive Gamma Values when C = 0.57339: ")
        for i in range(len(gamma_values) - 1):
            overlap = calculate_overlap(support_vectors_list[i], support_vectors_list[i + 1])
            print(f"Gamma: {gamma_values[i]:.5f} and {gamma_values[i + 1]:.5f}, Overlap: {overlap}")




