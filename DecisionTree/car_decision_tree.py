import pandas as pd
import numpy as np
from collections import Counter
from math import log2



columns_car = [
    ("buying", "categorical"),
    ("maint", "categorical"),
    ("doors", "categorical"),
    ("persons", "categorical"),
    ("lug_boot", "categorical"),
    ("safety", "categorical"),
    ("label", "categorical"),
]

target_variable = "label"

def attr_name(columns):
    return [attr_name for attr_name, _ in columns]


def attr_freq(labels):
    return Counter(labels)


def attr_prop(label_freq, num_rows):
    return [label_freq[label] / num_rows for label in label_freq]


def majority_error(data, num_rows, target_variable):
    label_freq = attr_freq(data[target_variable])
    label_prop = attr_prop(label_freq, num_rows)
    majority_error =  1 - max(label_prop)
    return majority_error


def gini_index(data, num_rows, target_variable):
    label_freq = attr_freq(data[target_variable])
    label_prop = attr_prop(label_freq, num_rows)
    gini_index = 1 - sum([p**2 for p in label_prop])
    return gini_index


def entropy(data, num_rows, target_variable):
    label_freq = attr_freq(data[target_variable])
    label_prop = attr_prop(label_freq, num_rows)
    entropy = -sum([p*log2(p) if p > 0 else 0 for p in label_prop])
    return entropy


def data_by_attr_val(data, attr, val):
    filtered_data = data[data[attr] == val]
    return [filtered_data, len(filtered_data)]


def information_gain(total_method_value, attr_prop, method_values):
    weighted_sum = sum([attr_prop[i] * method_values[i] for i in range(len(method_values))])
    information_gain = total_method_value - weighted_sum
    return information_gain


def max_gain_node(data, total_method_value, target_variable, method_function):
    information_gain_by_attr = {}

    for attr in data.columns:
        if attr != target_variable:
            attr_frequency = attr_freq(data[attr])
            attr_proportions = attr_prop(attr_frequency, len(data))

            method_values = [
                method_function(
                    *data_by_attr_val(data, attr, value),
                    target_variable
                )
                for value in attr_frequency
            ]

            information_gain_by_attr[attr] = information_gain(
                total_method_value, attr_proportions, method_values
            )

    if information_gain_by_attr:
        return max(information_gain_by_attr, key=information_gain_by_attr.get)


def decision_tree(data, target_variable, method, maximum_depth, depth=0):
    method_value = method(data, len(data), target_variable)
    node = max_gain_node(data, method_value, target_variable, method)

    tree = {node: {}}
    unique_values = np.unique(data[node])

    for i in unique_values:
        subset = data[data[node] == i]
        target_values = subset[target_variable].unique()

        if len(target_values) == 1:
            tree[node][i] = target_values[0]
        elif depth + 1 < maximum_depth:
            tree[node][i] = decision_tree(subset, target_variable, method, maximum_depth, depth + 1)

    return tree


def predict(tree, instance):
    if not isinstance(tree, dict):
        return tree

    root_node = next(iter(tree))
    feature_value = instance[root_node]

    if feature_value in tree[root_node]:
        return predict(tree[root_node][feature_value], instance)
    return None


def cal_error(tree, data, target_variable):
    correct = sum(1 for _, row in data.iterrows() if predict(tree, row) == row[target_variable])
    err = 1 - correct / len(data)
    return err


def car_decision_tree():
    train_df = pd.read_csv("DecisionTree/data/car/train.csv")
    train_df.columns = attr_name(columns_car)
    test_df = pd.read_csv("DecisionTree/data/car/test.csv")
    test_df.columns = attr_name(columns_car)

   
    method_mapping = {
        "majority_error": majority_error,
        "gini_index": gini_index,
        "entropy": entropy,
    }

    
    for method_name, method_func in method_mapping.items():
        print(f"\nMethod: {method_name}")

        # Modify the range to increase or decrease the depth of the tree
        for depth in range(1, 7):
            tree = decision_tree(train_df, target_variable, method_func, depth)

            train_error = cal_error(tree, train_df, target_variable)
            test_error = cal_error(tree, test_df, target_variable)

            print(f"Depth: {depth}, Train Error: {train_error:.3f}, Test Error: {test_error:.3f}")



if __name__ == "__main__":
    car_decision_tree()


