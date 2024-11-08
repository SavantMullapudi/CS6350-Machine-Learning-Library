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


learning_rate = 0.05
T = 10

for i in range(1, T + 1):
    for j in range(len(X_train)):
        xi = X_train[j]
        yi = y_train[j]

        if yi * np.dot(wt, xi) <= 0:
            wt = wt + learning_rate * yi * xi

test_errors = 0
for i in range(len(X_test)):
    if y_test[i] * np.dot(wt, X_test[i]) <= 0:
        test_errors += 1

average_error = test_errors / len(X_test)

print("Learned Weight Vector:", wt)
print("Average Prediction Error on the Test Data:", average_error)
