import numpy as np
import pandas as pd

column_names = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
train_data = pd.read_csv('ML HW 5/bank-note/train.csv', header=None, names=column_names)
test_data = pd.read_csv('ML HW 5/bank-note/test.csv', header=None, names=column_names)

X_train = train_data[['variance', 'skewness', 'curtosis', 'entropy']].values
y_train = train_data['label'].map({1: 1, 0: -1}).values
X_test = test_data[['variance', 'skewness', 'curtosis', 'entropy']].values
y_test = test_data['label'].map({1: 1, 0: -1}).values

def activation_fn(z):
    return 1 / (1 + np.exp(-z))

def activation_derivative(a):
    return a * (1 - a)

def init_weights(input_dim, hidden_dim, output_dim):
    np.random.seed(10)
    return {
        'w1': np.random.normal(0, 1, (input_dim, hidden_dim)),
        'w2': np.random.normal(0, 1, (hidden_dim, hidden_dim)),
        'w3': np.random.normal(0, 1, (hidden_dim, output_dim))
    }

def forward_pass(inputs, weights):
    h1_input = np.dot(inputs, weights['w1'])
    h1_output = activation_fn(h1_input)

    h2_input = np.dot(h1_output, weights['w2'])
    h2_output = activation_fn(h2_input)

    final_input = np.dot(h2_output, weights['w3'])
    final_output = activation_fn(final_input)

    return h1_output, h2_output, final_output

def backward_pass(inputs, target, output, h2_output, h1_output, weights, lr):
    error = target - output
    delta_out = error * activation_derivative(output)

    error_h2 = delta_out.dot(weights['w3'].T)
    delta_h2 = error_h2 * activation_derivative(h2_output)

    error_h1 = delta_h2.dot(weights['w2'].T)
    delta_h1 = error_h1 * activation_derivative(h1_output)

    weights['w3'] += lr * np.outer(h2_output, delta_out)
    weights['w2'] += lr * np.outer(h1_output, delta_h2)
    weights['w1'] += lr * np.outer(inputs, delta_h1)

def learning_rate_schedule(gamma0, d, t):
    return gamma0 / (1 + (gamma0 / d) * t)

def train_nn(X, y, hidden_dim, output_dim, gamma0, d, epochs):
    input_dim = X.shape[1]
    weights = init_weights(input_dim, hidden_dim, output_dim)
    objective_values = []
    
    t = 0  

    for epoch in range(epochs):
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

        for i in range(len(X)):
            t += 1
            inputs = X[i]
            target = y[i]

            h1_output, h2_output, final_output = forward_pass(inputs, weights)
            backward_pass(inputs, target, final_output, h2_output, h1_output, weights, learning_rate_schedule(gamma0, d, t))
            error = target - final_output
            objective_values.append(0.5 * np.square(error))

    return weights, objective_values

def test_nn(X, y, weights):
    predictions = []
    for i in range(len(X)):
        _, _, final_output = forward_pass(X[i], weights)
        predictions.append(1 if final_output[0] >= 0.5 else -1)

    accuracy = np.mean(predictions == y)
    return 1 - accuracy  

gamma0 = 0.1
d = 0.01
epochs = 50
hidden_layer_widths = [5, 10, 25, 50, 100]

train_errors = []
test_errors = []


for width in hidden_layer_widths:
    weights, objective_values = train_nn(X_train, y_train, width, 1, gamma0, d, epochs)

    train_error = test_nn(X_train, y_train, weights)
    test_error = test_nn(X_test, y_test, weights)

    train_errors.append(train_error)
    test_errors.append(test_error)

for i, width in enumerate(hidden_layer_widths):
    print(f"Width: {width}, Train Error: {train_errors[i] * 100:.2f}%, Test Error: {test_errors[i] * 100:.2f}%")