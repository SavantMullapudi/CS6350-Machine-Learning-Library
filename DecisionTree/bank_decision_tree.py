import pandas as pd
import numpy as np
from collections import Counter
from math import log2
from random import choice


columns_bank_dataset = [
    ("age", "numeric"),
    ("job", "categorical"),
    ("martial", "categorical"),
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

methods_bank_dataset = ["majority_error", "gini_index", "entropy"]


def attr_name(columns):
    return [attr_name for attr_name, _ in columns]

def attr_type(attribute):
    if isinstance(attribute, tuple):
        return attribute[1]

def attr_freq(labels):
    return Counter(labels)


def attr_prop(label_frequency, num_records):
    return [label_frequency[label] / num_records for label in label_frequency]


def majority_error(data, num_records, target_variable):
    label_frequency = attr_freq(data[target_variable])
    label_proportions = attr_prop(label_frequency, num_records)

    majority_error = 1 - max(label_proportions)
    return majority_error


def gini_index(data, num_records, target_variable):
    label_frequency = attr_freq(data[target_variable])
    label_proportions = attr_prop(label_frequency, num_records)
    modified_label_proportions = [(proportion) ** 2 for proportion in label_proportions]

    gini_index = 1 - sum(modified_label_proportions)
    return gini_index


def entropy(data, num_records, target_variable):
    label_frequency = attr_freq(data[target_variable])
    label_proportions = attr_prop(label_frequency, num_records)
    modified_label_proportions = [
        -((proportion) * log2(proportion)) for proportion in label_proportions
    ]

    entropy = sum(modified_label_proportions)
    return entropy


def data_attr_val(data, attribute, value):
    data_by_attribute_value = data[data[attribute] == value]
    return [data_by_attribute_value, data_by_attribute_value.shape[0]]


def information_gain(
    total_variant_value, attribute_proportions, method_values
):
    modified_products = [
        attribute_proportions[index] * method_values[index]
        for index in range(len(method_values))
    ]

    return total_variant_value - sum(modified_products)


def are_values_unknown(
    data,
    attribute,
):
    return "unknown" in data[attribute[0]].values

def maximum_gain_node_func(data, total_gini_index, target_variable):
    information_gain_by_attribute = dict()

    for index, attribute in enumerate(data.columns):
        if index != len(data.columns) - 1:
            attribute_frequency = attr_freq(data[attribute])
            attribute_proportions = attr_prop(
                attribute_frequency, data.shape[0]
            )

            attribute_gini_indices = [
                gini_index(
                    data_attr_val(data, attribute, value)[0],
                    data_attr_val(data, attribute, value)[1],
                    target_variable,
                )
                for value in attribute_frequency
            ]

            information_gain_by_attribute[attribute] = information_gain(
                total_gini_index,
                attribute_proportions,
                attribute_gini_indices,
            )

    if len(information_gain_by_attribute.keys()) > 0:
        maximum_information_gain = max(information_gain_by_attribute.values())
        maximum_information_gain_attributes = [
            attribute
            for attribute, gain in list(information_gain_by_attribute.items())
            if gain == maximum_information_gain
        ]

        return choice(maximum_information_gain_attributes)


def sub_tree(data, root_node, target_variable):
    sub_tree = dict()
    for attribute_value in attr_freq(data[root_node]):
        attribute_value_data = data[data[root_node] == attribute_value]
        unique_labels = set(attribute_value_data[target_variable])

        if len(unique_labels) == 1:
            sub_tree[attribute_value] = unique_labels.pop()
            data = data[data[root_node] != attribute_value]
        else:
            sub_tree[attribute_value] = "Not a leaf node"

    return [sub_tree, data]

def predict_values(tree, instance):
    if not isinstance(tree, dict):
        return [tree, "leaf-node"]
    else:
        root_node = next(iter(tree))
        feature_value = instance[root_node]
        if feature_value in tree[root_node]:
            return predict_values(tree[root_node][feature_value], instance)
        else:
            return [None, "leaf-node"]

def non_categorical_attr(attributes):
    numeric_attributes = []
    for attribute in attributes:
        if attr_type(attributes, attribute) == "numeric":
            numeric_attributes.append(attribute)

    return numeric_attributes


def categorical_data(data, attributes, are_unknowns_to_be_replaced):
    categorical_data = None
    for attribute in attributes:
        type_of_attribute = attr_type(attribute)
        if type_of_attribute == "numeric":
            median_value = median_of_attribute(data, attribute)
            categorical_data = assign_categorical_values(data, attribute, median_value)
        elif type_of_attribute == "categorical":
            if are_unknowns_to_be_replaced:
                are_any_values_unknown = are_values_unknown(data, attribute)
                if are_any_values_unknown:
                    conditions = [data[attribute[0]].eq("unknown")]
                    other_data = data[data[attribute[0]] != "unknown"]
                    
                    attribute_frequency = attr_freq(other_data[attribute[0]])

                    max_occurence = attribute_frequency.most_common(1)[0][0]
                    
                    options = [str(max_occurence)]

                    default_value = "unknown" 
                    data[attribute[0]] = np.select(conditions, options, default=default_value)
                    
                    categorical_data = data


    return categorical_data




def construct_tree(data, target_variable, method, maximum_depth, depth, tree=None):
    method_value = None
    if method == "gini_index":
        method_value = gini_index(data, data.shape[0], target_variable)
    elif method == "majority_error":
        method_value = majority_error(data, data.shape[0], target_variable)
    elif method == "entropy":
        method_value = entropy(data, data.shape[0], target_variable)

    maximum_gain_node = maximum_gain_node_func(
        data, method_value, target_variable
    )

    attribute_values = np.unique(data[maximum_gain_node])

    if tree is None:
        tree = dict()
        tree[maximum_gain_node] = dict()

    for value in attribute_values:
        sub_data = data[data[maximum_gain_node] == value].reset_index(drop=True)
        target_values = np.unique(sub_data[target_variable])

        if len(target_values) == 1:
            tree[maximum_gain_node][value] = target_values[0]
        else:
            depth += 1
            if depth < maximum_depth:
                tree[maximum_gain_node][value] = construct_tree(
                    sub_data, target_variable, method, maximum_depth, depth
                )

    return tree

def median_of_attribute(data, attribute):
    if data.shape[0] > 0 and attribute is not None:
        data = data[data[attribute[0]] != -1]
        return int(data[attribute[0]].median())


def assign_categorical_values(data, attribute, median):
    if data.shape[0] > 0 and attribute is not None and median is not None:
        conditions = [
            data[attribute[0]].gt(median),
            data[attribute[0]].le(median),
        ]
        options = [1, 0]
        data[attribute[0]] = np.select(conditions, options, default=0)

        return data

def cal_error(decision_tree, data, target_variable):
    num_correct_predictions, num_wrong_predictions = 0, 0

    for _, row in data.iterrows():
        result, _ = predict_values(decision_tree, row)

        if result == row[target_variable]:
            num_correct_predictions += 1
        else:
            num_wrong_predictions += 1

    error = 1 - (num_correct_predictions / (num_correct_predictions + num_wrong_predictions))

    return error

def bank_decision_tree():
    while True:
        try:
            are_unknowns_to_be_replaced = input(
                "\nDo you want to replace unknown values in the data?\nEnter 1 if you want to or 0 if you don't want to.\n"
            )
            break
        except:
            print("Let's try that one more time.")

    are_unknowns_to_be_replaced = are_unknowns_to_be_replaced == "1"


    train_df = "DecisionTree/data/bank/train.csv"
    test_df = "DecisionTree/data/bank/test.csv"
    training_data = pd.read_csv(train_df)
    training_data.columns = attr_name(columns_bank_dataset)
    updated_training_data = categorical_data(
        training_data, columns_bank_dataset, are_unknowns_to_be_replaced
    )

    test_data = pd.read_csv(test_df)
    test_data.columns = attr_name(columns_bank_dataset)
    updated_test_data = categorical_data(
        test_data, columns_bank_dataset, are_unknowns_to_be_replaced
    )

    methods = methods_bank_dataset 


    for val in methods:
        tree = construct_tree(
            training_data,
            target_variable,
            val,
            16,  
            0,
        )

        test_error = cal_error(tree, test_data, target_variable)
        train_error = cal_error(tree, training_data, target_variable)

    print("\nCalculating error scores for max depth 1 to 16.\n")



    for val in methods:
        print(f"Method: {val}")

        # Modify the range to increase or decrease the depth of the tree
        for depth in range(1, 17):
            tree = construct_tree(
                training_data,
                target_variable,
                val,
                depth,
                0,
            )

            test_error = cal_error(tree, test_data, target_variable)
            train_error = cal_error(tree, training_data, target_variable)

            print(f"\tDepth: {depth}, Train Error: {train_error:.3f}, Test Error: {test_error:.3f}")

        print("\n")




bank_decision_tree()