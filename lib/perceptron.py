import numpy as np
import matplotlib.pyplot as plt

class Perceptron(object):
    def __init__(self, eta=0.01, n_iterations=10):
        self.eta = eta
        self.n_iterations = n_iterations

    def fit(self, X, y):
        n_features = X.shape[1]
        self.w_ = np.zeros(1 + n_features)
        self.errors_ = []

        for _ in range (self.n_iterations):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


def process_iris(df):
    """
    Follow Machine Learning in Python book.
    Take only 100 first rows with 50 Iris-Setosa and 50 Iris-Versicolor
    data. Run Perceptron on the dataset.
    """
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    X = df.iloc[0:100, [0,2]].values
    p1 = plt.figure(1)
    plt.scatter(
        X[0:50, 0],
        X[0:50, 1],
        color='red',
        marker='o',
        label='setosa',
    )
    plt.scatter(
        X[50:100, 0],
        X[50:100, 1],
        color='blue',
        marker='x',
        label='versicolor',
    )
    p1.show()

    perceptron = Perceptron(eta=0.1, n_iterations=10)
    perceptron.fit(X,y)
    p2 = plt.figure(2)
    plt.plot(
        range(1, len(perceptron.errors_) + 1), perceptron.errors_, marker='o',
    )
    p2.show()

    input()