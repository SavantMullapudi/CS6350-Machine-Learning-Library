import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from collections import Counter

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

train_file = "data/bank/train.csv"
test_file = "data/bank/test.csv"

X_train, y_train = load_and_preprocess_data(train_file, columns_bank_dataset, target_variable)
X_test, y_test = load_and_preprocess_data(test_file, columns_bank_dataset, target_variable)

class Node_Tree:
    def __init__(self, attribute, attr_name, leaf_true, label, depth, info_gain, attr_ent_p, attr_val_p):
        self.attribute = attribute
        self.attr_name = attr_name
        self.children = {}
        self.leaf_true = leaf_true
        self.label = label
        self.depth = depth
        self.info_gain = info_gain
        self.attr_ent_p = attr_ent_p
        self.attr_val_p = attr_val_p

    def get_attribute(self):
        return self.attribute

    def add_child(self, child_node, attr_value):
        self.children[attr_value] = child_node
    
    def predict(self, x):
        if self.leaf_true:
            return self.label
        current_val = x[self.attribute]
        if current_val not in self.children.keys():
            return self.label
        return self.children[current_val].predict(x)


    def majority_error(self, X, y, attribute):
        values = set(X[attribute])
        val = sum([(X[attribute] == value).mean() *
            (1 - Counter(y[X[attribute] == value]).most_common(1)[0][1] / len(y[X[attribute] == value])) for value in values])
        return val

    def gini_index(self, X, y, attribute):
        values = set(X[attribute])
        g = 1
        for value in values:
            p = (X[attribute] == value).mean()
            g = g - p**2
        return g

class Construct_tree:
    def __init__(self, max_depth=np.inf):
        self.root = None
        self.depth = 0
        if max_depth < 1:
            print("max_depth never < 1. set max_depth =  1.")
            max_depth = 1
        self.max_depth = max_depth
        self.longest_path_len = 0

    def build_tree(self, X, Y, attribute_names, attribute_list=[], current_depth=0,
                   parent_info={"max_info_gain": None, "attribute_list[max_attribute]": None, "value": None}):
        if current_depth > self.longest_path_len:
            self.longest_path_len = current_depth
        if current_depth >= self.max_depth or len(attribute_list) == 0 or len(np.unique(Y)) == 1:
            vals, counts = np.unique(Y, return_counts=True)
            return Node_Tree(None, None, True, vals[np.argmax(counts)], current_depth,
                            parent_info["max_info_gain"], parent_info["attribute_list[max_attribute]"],
                            parent_info["value"])

        max_info_gain = -1
        max_attribute = None
        i = 0
        for attribute in attribute_list:
            info_gain, entropy_attribute, entropy_parent = self.info_gain(X, Y, attribute)
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                max_attribute = i
                entropy = entropy_parent
            i += 1

        vals, counts = np.unique(Y, return_counts=True)
        root = Node_Tree(attribute_list[max_attribute], attribute_names[attribute_list[max_attribute]],
                        False, vals[np.argmax(counts)], current_depth,
                        parent_info["max_info_gain"], parent_info["attribute_list[max_attribute]"],
                        parent_info["value"])

        attribute_values = np.unique(X[:, attribute_list[max_attribute]])
        new_attribute_list = np.delete(attribute_list, max_attribute)
        for value in attribute_values:
            indices = np.where(X[:, attribute_list[max_attribute]] == value)[0]
            if len(indices) == 0:
                root.add_child(Node_Tree(None, None, True, vals[np.argmax(counts)], current_depth + 1,
                                        max_info_gain, attribute_list[max_attribute], value), current_depth)
            else:
                parent_info = {
                    "max_info_gain": max_info_gain,
                    "attribute_list[max_attribute]": entropy,
                    "value": value
                }
                root.add_child(self.build_tree(X[indices], Y[indices], attribute_names, new_attribute_list,
                                               current_depth + 1, parent_info), value)
        return root

    def entropy_calc(self, counts):
        total = sum(counts)
        entropy_value = 0
        for element in counts:
            p = (element / total)
            if p != 0:
                entropy_value -= p * np.log2(p)
        return entropy_value

    def info_gain(self, X, Y, attribute):
        _, counts = np.unique(Y, return_counts=True)
        entropy_attribute = self.entropy_calc(counts)
        entropy_parent = 0
        distinct_attr_values = list(set(X[:, attribute]))
        for val in distinct_attr_values:
            indices = np.where(X[:, attribute] == val)[0]
            _, counts = np.unique(Y[indices], return_counts=True)
            entr = self.entropy_calc(counts)
            entropy_parent += (len(indices) / len(Y)) * entr
        info_gain = entropy_attribute - entropy_parent
        return info_gain, entropy_attribute, entropy_parent

    def fit(self, X, Y):
        attribute_names = list(range(X.shape[1])) 
        attribute_list = np.arange(X.shape[1])
        self.root = self.build_tree(X, Y, attribute_names, attribute_list, 0)

    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            predictions.append(self.root.predict(X[i]))
        return predictions

    def get_longest_path_len(self):
        return self.longest_path_len

    def get_root_attribute(self):
        if self.root:
            return self.root.get_attribute()
        return None



class Bagged_tree:
    def __init__(self, num_trees):
        self.num_trees = num_trees
        self.trees = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        for _ in range(self.num_trees):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap, y_bootstrap = X[indices], y[indices]

            dt_classifier = Construct_tree(max_depth=np.inf)
            dt_classifier.fit(X_bootstrap, y_bootstrap)
            self.trees.append(dt_classifier)

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for tree in self.trees:
            predictions += tree.predict(X)
        return np.sign(predictions)

range_tree = range(1, 501)
train_errors_bagging = []
test_errors_bagging = []

for num_trees in range_tree:

    bagged_trees_classifier = Bagged_tree(num_trees)
    bagged_trees_classifier.fit(X_train, y_train)


    y_train_pred = bagged_trees_classifier.predict(X_train)
    y_test_pred = bagged_trees_classifier.predict(X_test)


    train_error = 1 - accuracy_score(y_train, y_train_pred)
    test_error = 1 - accuracy_score(y_test, y_test_pred)

    train_errors_bagging.append(train_error)
    test_errors_bagging.append(test_error)


plt.figure(figsize=(12, 5))
plt.plot(range_tree, train_errors_bagging, label='Train Error', color='red', linestyle='--')
plt.plot(range_tree, test_errors_bagging, label='Test Error', color='black', linestyle='--')
plt.xlabel('Number of Trees', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.xticks([0, 10, 20, 30, 40, 50], [0, 100, 200, 300, 400, 500])
plt.title('Bagging Tree - Training and Test Errors vs. Number of Trees', fontsize=14)
plt.legend(loc='best')
plt.ylim(0.000, 0.0225)  
plt.yticks(np.arange(0.000, 0.225, 0.025)) 

plt.tight_layout()
plt.show()


