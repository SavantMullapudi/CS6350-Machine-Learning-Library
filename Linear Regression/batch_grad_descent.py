import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv('data/concrete/train.csv')
test_data = pd.read_csv('data/concrete/test.csv')

X_train = train_data.drop(columns=train_data.columns[-1]).values  
y_train = train_data.iloc[:, -1].values  

X_test = test_data.drop(columns=test_data.columns[-1]).values  
y_test = test_data.iloc[:, -1].values 

X_train_bias = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test_bias = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

def run_grad_descent(X, y, init_wts, lr, threshold, max_iters):
    wts = init_wts
    m = len(y)
    cost_history = []

    for iteration in range(max_iters):
        predictions = np.dot(X, wts)  
        residuals = predictions - y
        gradient = (1/m) * np.dot(X.T, residuals)  
        updated_wts = wts - lr * gradient
        cost = (1/(2*m)) * np.sum(residuals**2)
        cost_history.append(cost)

        if np.linalg.norm(updated_wts - wts) < threshold:
            break

        wts = updated_wts

    return wts, cost_history

initial_wts = np.zeros(X_train_bias.shape[1]) 
learning_rate = 0.01 
tolerance = 1e-6 
max_iterations = 100000  

optimal_wts, history_of_cost = run_grad_descent(X_train_bias, y_train, initial_wts, learning_rate, tolerance, max_iterations)

plt.plot(history_of_cost, color='orange')  
plt.xlabel('Iterations', fontsize=12) 
plt.ylabel('Cost', fontsize=12)  
plt.title('Cost vs Iterations Comparison', fontsize=14)  
plt.grid(False)  
plt.show()

y_pred_test = np.dot(X_test_bias, optimal_wts)  

mse_test = (1/(2*len(y_test))) * np.sum((y_pred_test - y_test)**2)

print("Optimized Wts:", optimal_wts)
print("Learning Rate:", learning_rate)
print("Test Data MSE:", mse_test)


