import pandas as pd

from lib.perceptron import process_iris

DATA_PATH = '_data/iris.data'

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    print (df.tail())
    process_iris(df)