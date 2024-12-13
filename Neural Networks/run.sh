#!/bin/sh

echo "back propogation on bank-note dataset"
python back_proportion_algo.py

echo "Results for stochastic gradient"
python stoc_grad_algo.py

echo "Results for stochastic gradient with zero initial weights"
python stoc_grad_wt0.py

