#!/bin/sh

echo "Result of 2a"
python primal_domain_svm.py

echo "Result of 2b"
python primal_domain_svm2.py

echo "Result of 2c"
python primal_svm_diff.py

echo "Result of 3a"
python dual_svm.py

echo "Result of 3b"
python dual_kernel_svm.py

echo "Result of 3c"
python support_vector_svm.py

echo "Result of 3d"
python perceptron_kernel_svm.py
