import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

def gaussian_kernel(x1, x2, gamma):
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

def kernel_perceptron(X_train, y_train, X_test, y_test, gamma, max_iter=100):
    n_samples, n_features = X_train.shape
    alpha = np.zeros(n_samples) 
    b = 0  

    for _ in range(max_iter):
        for i in range(n_samples):
            f_x = sum(alpha[j] * y_train[j] * gaussian_kernel(X_train[j], X_train[i], gamma) for j in range(n_samples)) + b
            if y_train[i] * f_x <= 0:
                alpha[i] += 1
                b += y_train[i]

    def predict(X):
        predictions = []
        for x in X:
            f_x = sum(alpha[j] * y_train[j] * gaussian_kernel(X_train[j], x, gamma) for j in range(n_samples)) + b
            predictions.append(1 if f_x >= 0 else -1)
        return np.array(predictions)

    y_train_pred = predict(X_train)
    y_test_pred = predict(X_test)

    train_error = 1 - accuracy_score(y_train, y_train_pred)
    test_error = 1 - accuracy_score(y_test, y_test_pred)

    return train_error, test_error


def main():
    X, y = make_classification(n_samples=500, n_features=2, n_classes=2, n_informative=2, n_redundant=0, random_state=42)
    y = np.where(y == 0, -1, 1) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    gamma_values = [0.1, 0.5, 1, 5, 100]
    
    print("Kernel Perceptron Results:")
    print("Gamma\tTrain Error\tTest Error")
    perceptron_results = []
    for gamma in gamma_values:
        train_error, test_error = kernel_perceptron(X_train, y_train, X_test, y_test, gamma)
        perceptron_results.append((gamma, train_error, test_error))
        print(f"{gamma}\t{train_error:.4f}\t\t{test_error:.4f}")

    print("\nNonlinear SVM Results:")
    print("Gamma\tTrain Error\tTest Error")
    svm_results = []
    for gamma in gamma_values:
        svm = SVC(kernel='rbf', gamma=gamma, C=1.0)
        svm.fit(X_train, y_train)

        train_error = 1 - svm.score(X_train, y_train)
        test_error = 1 - svm.score(X_test, y_test)
        svm_results.append((gamma, train_error, test_error))
        print(f"{gamma}\t{train_error:.4f}\t\t{test_error:.4f}")

if __name__ == "__main__":
    main()
