import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from collections import Counter

col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
train_df = pd.read_csv("train.csv", names=col_names)
X_train = train_df.drop('label', axis=1).values
y_train = train_df['label'].values

test_df = pd.read_csv("test.csv", names=col_names)
X_test = test_df.drop('label', axis=1).values
y_test = test_df['label'].values

class DecisionTree:
    def __init__(self, attr_index, attr_Name, is_leaf, label, depth, info_gain, entropy, parent_val):
        self.attr_index = attr_index
        self.attr_Name = attr_Name
        self.children = {}
        self.is_leaf = is_leaf
        self.label = label
        self.depth = depth
        self.info_gain = info_gain
        self.entropy = entropy
        self.parent_val = parent_val

    def add_child(self, child_node, attr_value):
        self.children[attr_value] = child_node

    def predict(self, row):
        if self.is_leaf:
            return self.label
        current_val = row[self.attr_index]
        if current_val in self.children:
            return self.children[current_val].predict(row)
        else:
            return self.label

    def majority_error(self, X, y, attr_index):
        val = np.unique(X[:, attr_index])
        err_sum = 0.0
        
        for i in val:
            indices = np.where(X[:, attr_index] == val)[0]
            if len(ind) == 0:
                continue

            class_counts = Counter(y[ind])
            most_common_count = class_counts.most_common(1)[0][1]  
            me_value = 1 - (most_common_count / len(ind)) 

            err_sum = err_sum + (len(ind) / len(y)) * me_value
            
        return err_sum

    def gini_index(self, X, y, attr_index):
        values = set(X[attr_index])
        gini = 1
        for value in values:
            p = (X[attr_index] == value).mean()
            gini = gini - (p**2)
        return gini

    def display_node(self, indent=""):
        details = [
            f"{indent}Depth: {self.depth}",
            f"{indent}Attribute Name: {self.attr_Name}",
            f"{indent}Information Gain: {self.info_gain}",
            f"{indent}Entropy: {self.entropy}",
            f"{indent}Parent's Value: {self.parent_val}",
            f"{indent}Predicted Label: {self.label}"
        ]
        for val in details:
            print(val)
        for child in self.children.values():
            child.display_node(indent + "  ")

class DecisionTreeClassifier:
    def __init__(self, max_depth=np.inf):
        self.root = None
        self.tree_depth = 0
        self.max_depth = max_depth
        self.max_path_length = 0

    def build_tree(self, X, y, features, available_features=None, current_depth=0,
                   parent_info={"best_gain": None, "best_feature": None, "parent_value": None}):
        if available_features is None:
            available_features = list(range(len(features)))
        if current_depth > self.max_path_length:
            self.max_path_length = current_depth
        if (current_depth >= self.max_depth) or (len(available_features) == 0) or (len(np.unique(y)) == 1):
            classes, counts = np.unique(y, return_counts=True)
            majority_class = classes[np.argmax(counts)]
            return DecisionTree(
                attr_index=None, 
                attr_Name=None, 
                is_leaf=True, 
                label=majority_class, 
                depth=current_depth,
                info_gain=parent_info["best_gain"],
                entropy=parent_info["best_feature"],
                parent_val=parent_info["parent_value"]
            )
        best_gain = -1
        best_feature_index = None
        best_entropy = None
        for i, attr_index in enumerate(available_features):
            info_gain, feature_entropy = self.calculate_information_gain(X, y, attr_index)
            if info_gain > best_gain:
                best_gain = info_gain
                best_feature_index = attr_index
                best_entropy = feature_entropy
        classes, counts = np.unique(y, return_counts=True)
        majority_class = classes[np.argmax(counts)]
        node = DecisionTree(
            attr_index=best_feature_index, 
            attr_Name=features[best_feature_index], 
            is_leaf=False, 
            label=majority_class, 
            depth=current_depth,
            info_gain=best_gain,
            entropy=parent_info["best_feature"],
            parent_val=parent_info["parent_value"]
        )
        feature_values = np.unique(X[:, best_feature_index])
        remaining_features = np.delete(available_features, i)
        for value in feature_values:
            subset_indices = np.where(X[:, best_feature_index] == value)[0]
            if len(subset_indices) == 0:
                node.add_child(
                    DecisionTree(
                        attr_index=None, 
                        attr_Name=None, 
                        is_leaf=True, 
                        label=majority_class, 
                        depth=current_depth + 1,
                        info_gain=best_gain,
                        entropy=best_entropy,
                        parent_val=value
                    ), 
                    value
                )
            else:
                child_info = {
                    "best_gain": best_gain,
                    "best_feature": best_entropy,
                    "parent_value": value
                }
                node.add_child(
                    self.build_tree(
                        X[subset_indices], 
                        y[subset_indices], 
                        features, 
                        remaining_features, 
                        current_depth + 1, 
                        child_info
                    ), 
                    value
                )
        return node

    def calculate_entropy(self, counts):
        total = sum(counts)
        entropy_value = 0
        for count in counts:
            prob = count / total
            if prob != 0:
                entropy_value -= prob * np.log2(prob)
        return entropy_value

    def calculate_information_gain(self, X, y, attr_index):
        total_entropy = self.calculate_entropy(np.unique(y, return_counts=True)[1])
        weighted_entropy = 0
        feature_values = np.unique(X[:, attr_index])
        for value in feature_values:
            subset_indices = np.where(X[:, attr_index] == value)[0]
            _, subset_counts = np.unique(y[subset_indices], return_counts=True)
            weighted_entropy += (len(subset_indices) / len(y)) * self.calculate_entropy(subset_counts)
        info_gain = total_entropy - weighted_entropy
        return info_gain, total_entropy

    def fit(self, X, y):
        features = list(range(X.shape[1]))
        self.root = self.build_tree(X, y, features)

    def predict(self, X):
        return [self.root.predict(row) for row in X]

    def get_longest_path_length(self):
        return self.max_path_length

    def display_tree(self):
        if self.root:
            self.root.display_node()

results = {'information_gain (IG)': [], 'majority_error (ME)': [], 'gini_index (GI)': []}



for max_depth in range(1, 7):  # Modify the range to vary the depth of the tree

    for criterion in ['information_gain (IG)', 'majority_error (ME)', 'gini_index (GI)']:
        model = DecisionTreeClassifier(max_depth=max_depth)
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        if criterion not in results:
            results[criterion] = [] 

        results[criterion].append((1 - train_acc, 1 - test_acc))

print('Method  \t\tDepth \tTrain Error    \tTest Error')
for criterion, errors in results.items():
    for depth, (train_error, test_error) in enumerate(errors, start=1):
        print(f'{criterion: <15} \tDepth {depth}: \t{train_error:.3f} \t{test_error:.3f}')

