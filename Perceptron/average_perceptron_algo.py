import pandas as pd
import numpy as np

column_names = ['variance', 'skewness', 'curtosis', 'entropy', 'label']

train_data = pd.read_csv('bank-note/train.csv', header=None, names=column_names)
test_data = pd.read_csv('bank-note/test.csv', header=None, names=column_names)

X_train = train_data[['variance', 'skewness', 'curtosis', 'entropy']].values  
y_train = train_data['label'].values               
X_test = test_data[['variance', 'skewness', 'curtosis', 'entropy']].values    
y_test = test_data['label'].values 

n = X_train.shape[1]
wt = np.zeros(n)
a = np.zeros(n)

learning_rate = 0.05
T = 10

weight_sums = []

for epoch in range(1, T+1):
    for j in range(len(X_train)):
        xi = X_train[j]
        yi = y_train[j]

        if yi * np.dot(wt, xi) <= 0:
            wt += learning_rate * yi * xi

    weight_sums.append(wt.copy())

a = np.mean(weight_sums, axis=0)

test_errors = np.sum([1 for i in range(len(X_test)) if y_test[i] * np.dot(a, X_test[i]) <= 0])

average_error = test_errors / len(X_test)

print("Learned Weight Vector (Average Weight Vector):", a)
print("Average Prediction Error on Test Data:", average_error)
