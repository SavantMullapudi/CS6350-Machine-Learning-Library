import numpy as np
import pandas as pd

column_names = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
train_data = pd.read_csv('bank-note/train.csv', header=None, names=column_names)
test_data = pd.read_csv('bank-note/test.csv', header=None, names=column_names)

X_train = train_data[['variance', 'skewness', 'curtosis', 'entropy']].values
y_train = train_data['label'].map({1: 1, 0: 0}).values
X_test = test_data[['variance', 'skewness', 'curtosis', 'entropy']].values
y_test = test_data['label'].map({1: 1, 0: 0}).values

def activation_fn(z):
    return 1 / (1 + np.exp(-z))

def learning_rate_schedule(gamma0, d, t):
    return gamma0 / (1 + (gamma0 / d) * t)

def objective_function(y, y_pred, w, variance):
    likelihood_term = -np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    prior_term = 0.5 / variance * np.sum(w**2)
    return likelihood_term + prior_term

def gradient(y, y_pred, X, w, variance):
    return -np.dot(X.T, (y - y_pred)) + (1 / variance) * w

def train_logistic_regression(X_train, y_train, variance, gamma0, d, epochs):
    input_dim = X_train.shape[1]
    w = np.zeros(input_dim + 1)  
    objective_values = []
    
    t = 0

    for epoch in range(epochs):
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        for i in range(len(X_train)):
            t += 1
            inputs = np.append(X_train[i], 1)  
            target = y_train[i]

            y_pred = activation_fn(np.dot(w, inputs))

            objective_values.append(objective_function(target, y_pred, w, variance))

            grad = gradient(target, y_pred, inputs, w, variance)
            lr = learning_rate_schedule(gamma0, d, t)
            w -= lr * grad

    return w, objective_values

def evaluate(X, y, w):
    X_with_bias = np.column_stack([X, np.ones(len(X))])  
    y_pred = activation_fn(np.dot(X_with_bias, w))
    y_pred_class = (y_pred >= 0.5).astype(int)
    error = np.mean(y_pred_class != y)
    return error

variances = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
gamma0 = 0.1
d = 0.01
epochs = 100

train_errors = []
test_errors = []

for variance in variances:
    weights, objective_values = train_logistic_regression(X_train, y_train, variance, gamma0, d, epochs)

    train_error = evaluate(X_train, y_train, weights)
    train_errors.append(train_error)

    test_error = evaluate(X_test, y_test, weights)
    test_errors.append(test_error)

print("\nSummary of Results:")
for i, variance in enumerate(variances):
    print(f"Variance: {variance}, Train Error: {train_errors[i] * 100:.2f}%, Test Error: {test_errors[i] * 100:.2f}%")
