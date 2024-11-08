import pandas as pd
import numpy as np

column_names = ['variance', 'skewness', 'curtosis', 'entropy', 'label']

train_data = pd.read_csv('bank-note/train.csv', header=None, names=column_names)
test_data = pd.read_csv('bank-note/test.csv', header=None, names=column_names)

X_train = train_data[['variance', 'skewness', 'curtosis', 'entropy']].values  
y_train = train_data['label'].values               
X_test = test_data[['variance', 'skewness', 'curtosis', 'entropy']].values    
y_test = test_data['label'].values 

def voted_algo(X_train, y_train, T, initial_step_rate):
    n_samples, n_features = X_train.shape
    wt = np.zeros(n_features)  
    dist_wt_vectors = [] 
    count_arr = [] 

    step_rate = initial_step_rate 

    for i in range(T):
        mistakes = 0
        for j, x in enumerate(X_train):
            y_pred = np.sign(np.dot(wt, x))
            if y_pred == 0:
                y_pred = -1  

            if y_pred * y_train[j] <= 0:
                wt = wt + step_rate * y_train[j] * x  
                mistakes += 1  

        step_rate = initial_step_rate / (1 + i * 0.1)  

        dist_wt_vectors.append(wt.copy())
        count_arr.append(n_samples - mistakes)

    return dist_wt_vectors, count_arr


initial_learning_rate = 0.05  
T = 10  

dist_wt_vectors, count_arr = voted_algo(X_train, y_train, T, initial_learning_rate)

test_errors = []

for k in range(len(dist_wt_vectors)):
    wt = dist_wt_vectors[k]
    errors = sum(
        1 for i, x in enumerate(X_test)
        if np.sign(np.dot(wt, x)) * y_test[i] <= 0  
    )
    new_error = errors / len(X_test)
    test_errors.append(new_error)

average_test_error = np.mean(test_errors)

for i, (wt, count) in enumerate(zip(dist_wt_vectors, count_arr)):
    print(f"Weight Vector {i + 1}: {wt}, Correct Count: {count}")

print(f"Average Test Error: {average_test_error:.2f}")
