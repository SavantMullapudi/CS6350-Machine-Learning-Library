#!/bin/sh

echo "Adaboost Algorithm"
python3 adaboost_algo.py

echo "Bagged Tree Algorithm"
python3 baggedtree_algo.py

echo "Bias Variance Algorithm"
python3 biasvariance_algo.py

echo "Random Forest Algorithm"
python3 randomforest_algo.py

echo "Comparison for all three (Algorithm)"
python3 single_rf_whole_forest.py
