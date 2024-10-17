import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

train_data = "DecisionTree/data/bank/train.csv"
test_data = "DecisionTree/data/bank/test.csv"

columns_bank_dataset = [
    ("age", "numeric"),
    ("job", "categorical"),
    ("marital", "categorical"),
    ("education", "categorical"),
    ("default", "categorical"),
    ("balance", "numeric"),
    ("housing", "categorical"),
    ("loan", "categorical"),
    ("contact", "categorical"),
    ("day", "numeric"),
    ("month", "categorical"),
    ("duration", "numeric"),
    ("campaign", "numeric"),
    ("pdays", "numeric"),
    ("previous", "numeric"),
    ("poutcome", "categorical"),
    ("y", "categorical"),
]
target_variable = "y"

def load_and_preprocess_data(file, columns_structure, target_variable):
    dtype_dict = {}
    column_headers = [col[0] for col in columns_structure]

    for col, col_type in columns_structure:
        if col_type == "numeric":
            dtype_dict[col] = float
        else:
            dtype_dict[col] = str 

    df = pd.read_csv(file, names=column_headers, dtype=dtype_dict)

    df[target_variable] = df[target_variable].apply(lambda x: 1 if x == 'yes' else -1)
    
    X = df.drop(target_variable, axis=1).values
    y = df[target_variable].values

    return X, y

X_train, y_train = load_and_preprocess_data(train_data, columns_bank_dataset, target_variable)
X_test, y_test = load_and_preprocess_data(test_data, columns_bank_dataset, target_variable)

class Stump_DecisionTree:
    def __init__(self, attr, limit, less_than, greater_than):
        self.attr = attr
        self.limit = limit
        self.less_than = less_than
        self.greater_than = greater_than

    def predict(self, x):
        if x[self.attr] <= self.limit:
            return self.less_than
        else:
            return self.greater_than

    @staticmethod
    def info_gain_calc(X, y, attr, limit, weights):
        n = len(y)
        left_ind = X[:, attr] <= limit
        right_ind = X[:, attr] > limit

        wt_l = np.sum(weights[left_ind])
        rt_l = np.sum(weights[right_ind])

        if wt_l == 0 or rt_l == 0:
            return 0

        entropy_lt = -np.sum(weights[left_ind] * np.log2(weights[left_ind] / wt_l))
        entropy_rt = -np.sum(weights[right_ind] * np.log2(weights[right_ind] / rt_l))

        entropy_tot = (wt_l / n) * entropy_lt + (rt_l / n) * entropy_rt
        return entropy_tot

    @staticmethod
    def find_best_split(X, y, weights):
        num_features = X.shape[1]
        best_limit = None
        best_attr = None
        min_entropy = float('inf')

        for attr in range(num_features):
            unique_values = np.unique(X[:, attr])
            limits = (unique_values[:-1] + unique_values[1:]) / 2

            for limit in limits:
                entropy = Stump_DecisionTree.info_gain_calc(X, y, attr, limit, weights)

                if entropy < min_entropy:
                    min_entropy = entropy
                    best_limit = limit
                    best_attr = attr

        return best_attr, best_limit

class AdaBoost:
    def __init__(self, num_iterations, learning_rate=0.5):
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.alphas = []
        self.stumps = []
        self.training_errors = []  
        self.test_errors = []  
        self.stump_errors = []  

    def fit(self, X, y):
        X = self.convert_to_numeric(X)
        y = y.astype(float)
        n = len(y)
        weights = np.ones(n) / n

        for t in range(self.num_iterations):
            stump = self.train_stump(X, y, weights)
            predictions = np.array([stump.predict(x) for x in X])
            predictions = predictions.astype(float)

            error = np.sum(weights * (predictions != y))

            if error == 0:
                alpha = 1.0
            else:
                alpha = self.learning_rate * np.log((1 - error) / max(error, 1e-10))
            self.alphas.append(alpha)

            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)

            self.stumps.append(stump)

            train_predictions = self.predict(X_train)
            test_predictions = self.predict(X_test)
            train_error = 1 - accuracy_score(y_train, train_predictions)
            test_error = 1 - accuracy_score(y_test, test_predictions)
            self.training_errors.append(train_error)
            self.test_errors.append(test_error)
            self.stump_errors.append(error / n)

    def convert_to_numeric(self, X):
        for i in range(X.shape[1]):
            col = X[:, i]

            if not np.issubdtype(col.dtype, np.number):
                col = pd.to_numeric(col, errors='coerce')
                col[np.isnan(col)] = 0

            X[:, i] = col

        return X

    def train_stump(self, X, y, weights):
        best_error = float('inf')
        best_stump = None

        for _ in range(2):
            attr, limit = Stump_DecisionTree.find_best_split(X, y, weights)
            for less_than in [-1, 1]:
                predictions = np.where(X[:, attr] <= limit, less_than, -less_than)
                error = np.sum(weights * (predictions != y))
                if error < best_error:
                    best_error = error
                    best_stump = Stump_DecisionTree(attr=attr, limit=limit,
                                                   less_than=less_than, greater_than=-less_than)

        return best_stump

    def predict(self, X):
        n = X.shape[0]
        predictions = np.zeros(n)

        for alpha, stump in zip(self.alphas, self.stumps):
            predictions += alpha * np.array([stump.predict(x) for x in X])

        return np.sign(predictions)

num_iterations = 500
learning_rate = 0.50  
adaboost = AdaBoost(num_iterations, learning_rate)


adaboost.fit(X_train, y_train)

train_predictions = adaboost.predict(X_train)
test_predictions = adaboost.predict(X_test)
train_error = 1 - accuracy_score(y_train, train_predictions)
test_error = 1 - accuracy_score(y_test, test_predictions)

print("Training Error:", train_error)
print("Test Error:", test_error)



plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_iterations + 1), adaboost.training_errors, label='Training Error', color='red', marker='o', linestyle='--')
plt.plot(range(1, num_iterations + 1), adaboost.test_errors, label='Test Error', color='black', marker='o', linestyle='-')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.title('Training and Test Errors vs. Iteration', fontsize=14)
plt.legend(loc='best')
plt.ylim(0.460, 0.480)  
plt.yticks(np.arange(0.460, 0.481, 0.004)) 


plt.subplot(1, 2, 2)
plt.plot(range(1, num_iterations + 1), adaboost.stump_errors, label='Decision Stump Error', color='black', marker='o', linestyle='-')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.title('Decision Stump Errors vs. Iteration', fontsize=14)
plt.legend(loc='best')
plt.ylim(0.000095, 0.000101)  
plt.yticks(np.arange(0.000095, 0.000101, 0.000001)) 



plt.tight_layout()
plt.show()





