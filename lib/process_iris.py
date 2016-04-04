import numpy as np
import matplotlib.pyplot as plt


def process_iris(df):
    """
    Follow Machine Learning in Python book.
    Take only 100 first rows with 50 Iris-Setosa and 50 Iris-Versicolor
    data.
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

    input()

    return (X, y)