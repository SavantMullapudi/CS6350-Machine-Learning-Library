import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train_data = pd.read_csv('DecisionTree/data/concrete/train.csv')
test_data = pd.read_csv('DecisionTree/data/concrete/test.csv')

X_train = train_data.drop(columns=train_data.columns[-1]).values  
y_train = train_data.iloc[:, -1].values  

X_test = test_data.drop(columns=test_data.columns[-1]).values  
y_test = test_data.iloc[:, -1].values 


X_train_bias = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test_bias = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

def run_stochastic_grad_descent(X, y, init_wts, lr, max_iters):
    wts = init_wts
    m = len(y)
    cost_history = []

    for iteration in range(max_iters):
        for i in range(m):
            random_index = np.random.randint(m)
            X_i = X[random_index:random_index + 1]
            y_i = y[random_index:random_index + 1]

            predictions = np.dot(X_i, wts)
            residuals = predictions - y_i
            
            gradient = X_i.T * residuals
            wts -= lr * gradient.flatten()
        
        predictions_all = np.dot(X, wts)
        cost = (1/(2*m)) * np.sum((predictions_all - y)**2)
        cost_history.append(cost)

    return wts, cost_history

initial_wts = np.zeros(X_train_bias.shape[1]) 
learning_rate = 0.0025  
max_iterations = 1000  

optimal_wts_sgd, history_of_cost_sgd = run_stochastic_grad_descent(X_train_bias, y_train, initial_wts, learning_rate, max_iterations)

plt.plot(history_of_cost_sgd, color='orange')  
plt.xlabel('Iterations', fontsize=12) 
plt.ylabel('Cost', fontsize=12)  
plt.title('Cost vs Iterations (SGD)', fontsize=14)  
plt.grid(False)  
plt.show()

# Evaluate the model on test data
y_pred_test_sgd = np.dot(X_test_bias, optimal_wts_sgd)  
mse_test_sgd = (1/(2*len(y_test))) * np.sum((y_pred_test_sgd - y_test)**2)

print("Optimized Weights (SGD):", optimal_wts_sgd)
print("Learning Rate (SGD):", learning_rate)
print("Test Data MSE (SGD):", mse_test_sgd)
