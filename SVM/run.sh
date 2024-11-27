#!/bin/sh

echo "Result of primal_domain_svm"
python primal_domain_svm.py

echo "Result of primal_domain_svm2"
python primal_domain_svm2.py

echo "Result of primal_svm_diff"
python primal_svm_diff.py

echo "Result of dual_svm"
python dual_svm.py

echo "Result of dual_kernel_svm"
python dual_kernel_svm.py

echo "Result of support_vector_svm"
python support_vector_svm.py

echo "Result of perceptron_kernel_svm"
python perceptron_kernel_svm.py
