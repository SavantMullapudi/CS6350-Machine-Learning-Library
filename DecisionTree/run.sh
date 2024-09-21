#!/bin/bash

echo "Car dataset train / test errors"
python3 car_decision_tree.py

echo "Bank dataset train / test errors. Press 1 for unknowns to be replaced and 0 for unknowns to remain same"
python3 bank_decision_tree.py

echo "Bank dataset train / test errors. Press 1 for unknowns to be replaced and 0 for unknowns to remain same"
python3 bank_decision_tree.py
