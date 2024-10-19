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

optimal_w = np.linalg.inv(X_train_bias.T.dot(X_train_bias)).dot(X_train_bias.T).dot(y_train)

print("Optimal Weight Vector:", optimal_w)

predictions_test = X_test_bias.dot(optimal_w)

m_test = len(y_test)
cost_test = (1 / (2 * m_test)) * np.sum((predictions_test - y_test) ** 2)

print("Cost Function of test data:", cost_test)
