import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
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
    def __init__(self, attribute, attribute_name, is_leaf, value, depth, info_gain, split_attr, split_value):
        self.attribute = attribute
        self.attribute_name = attribute_name
        self.is_leaf = is_leaf
        self.value = value
        self.depth = depth
        self.info_gain = info_gain
        self.split_attr = split_attr
        self.split_value = split_value
        self.children = {}

    def add_child(self, child_node, value):
        self.children[value] = child_node

    def predict(self, x):
        if self.is_leaf:
            return self.value
        value = x[self.attribute]
        if value in self.children:
            return self.children[value].predict(x)
        else:
            return self.value

    def get_attribute(self):
        return self.attribute


class Construct_tree:
    def __init__(self, max_depth=np.inf):
        self.root = None
        self.depth = 0
        if max_depth < 1:
            print("max_depth never < 1. set max_depth =  1.")
            max_depth = 1
        self.max_depth = max_depth
        self.longest_path_len = 0

    def fit(self, X, Y):  
        attribute_names = list(range(X.shape[1])) 
        attribute_list = np.arange(X.shape[1])
        self.root = self.build_tree(X, Y, attribute_names, attribute_list, 0)

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
            i = i+1

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

    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            predictions.append(self.root.predict(X[i]))
        return predictions


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


def bias_var(predictions, ground_truth):
    bias = np.mean(predictions) - ground_truth
    variance = np.var(predictions)
    return bias, variance


def outcome_exp(X_train, y_train, X_test, y_test, num_iterations=100, num_btrees=500):
    stree_biases = []
    stree_var = []
    btree_biases = []
    btree_var = []

    for _ in range(num_iterations):
        n_samples = X_train.shape[0]
        samp_ind = np.random.choice(n_samples, size=1000, replace=False)
        samp_X_train, samp_y_train = X_train[samp_ind], y_train[samp_ind]

        classifier_btrees = Bagged_tree(num_btrees)
        classifier_btrees.fit(samp_X_train, samp_y_train)

        stree_pred = np.array([tree.predict(X_test) for tree in classifier_btrees.trees])
        avg_stree_pred = np.mean(stree_pred, axis=0)
        stree_bias, stree_variance = bias_var(avg_stree_pred, y_test)
        stree_biases.append(stree_bias)
        stree_var.append(stree_variance)

        btree_preds = classifier_btrees.predict(X_test)
        btree_bias, btree_variance = bias_var(btree_preds, y_test)
        btree_biases.append(btree_bias)
        btree_var.append(btree_variance)

    avg_stree_bias = np.mean(stree_biases)
    avg_stree_variance = np.mean(stree_var)
    avg_btree_bias = np.mean(btree_biases)
    avg_btree_variance = np.mean(btree_var)

    return avg_stree_bias, avg_stree_variance, avg_btree_bias, avg_btree_variance

stree_bias, stree_variance, btree_bias, btree_variance = outcome_exp(
    X_train, y_train, X_test, y_test
)

print(f"Single Tree Bias: {stree_bias}, Single Tree Variance: {stree_variance}")
print(f"Bagged Tree Bias: {btree_bias}, Bagged Tree Variance: {btree_variance}")

