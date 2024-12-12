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

def init_weights(input_dim, hidden1_dim, hidden2_dim, output_dim):
    np.random.seed(10)
    return {
        'w1': np.random.normal(0, 0.5, (input_dim, hidden1_dim)),
        'w2': np.random.normal(0, 0.5, (hidden1_dim, hidden2_dim)),
        'w3': np.random.normal(0, 0.5, (hidden2_dim, output_dim))
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


def train_nn(X, y, hidden1_dim, hidden2_dim, output_dim, lr, epochs):
    input_dim = X.shape[1]
    weights = init_weights(input_dim, hidden1_dim, hidden2_dim, output_dim)

    for _ in range(epochs):
        for i in range(len(X)):
            inputs = X[i]
            target = y[i]

            h1_output, h2_output, final_output = forward_pass(inputs, weights)
            backward_pass(inputs, target, final_output, h2_output, h1_output, weights, lr)

    return weights

def test_nn(X, y, weights):
    predictions = []
    for i in range(len(X)):
        _, _, final_output = forward_pass(X[i], weights)
        predictions.append(1 if final_output[0] >= 0.5 else -1)

    accuracy = np.mean(predictions == y)
    return accuracy


hidden1_dim = 4  
hidden2_dim = 4  
output_dim = 1
learning_rate = 0.003  
epochs = 100

# Train and Test
tuned_weights = train_nn(X_train, y_train, hidden1_dim, hidden2_dim, output_dim, learning_rate, epochs)
final_accuracy = test_nn(X_test, y_test, tuned_weights)

print(f"Accuracy on test set: {final_accuracy * 100:.2f}%")
