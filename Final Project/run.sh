#!/bin/sh

echo "Income > 50K probability predictions using decission_tree "
python3 ml_comp_decision_tree.py

echo "Income > 50K probability predictions using naive_bayes "
python3 ml_comp_naive_bayes.py

echo "Income > 50K probability predictions using adaboost "
python3 ml_comp_adaboost.py

echo "Income > 50K probability predictions using k_nearest_neighbors "
python3 ml_comp_KNN.py

echo "Income > 50K probability predictions using support_vector_machine "
python3 ml_comp_svm.py

echo "Income > 50K probability predictions using logistic_regression "
python3 ml_comp_logistic_regression.py

