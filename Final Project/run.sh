#!/bin/sh

echo "Income>50K probability predictions using decissionTree "
python3 incomePredictionDecissionTree.py

echo "Income>50K probability predictions using Random Forest "
python3 incomePredictionRandomForest.py

echo "Income>50K probability predictions using Adaboost "
python3 incomePredictionAdaboost.py

echo "Income>50K probability predictions using Perceptron "
python3 incomePredictionPerceptron.py

echo "Income>50K probability predictions using SVM "
python3 incomePredictionSVM.py

echo "Income>50K probability predictions using Neural Network "
python3 incomePredictionNeuralNetwork.py
