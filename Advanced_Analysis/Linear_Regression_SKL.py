import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso


# Data preparation.

def normalize(df, on=None, ignore=(), offset=1e-5):
    on = on if on is not None else df
    for col in set(df.columns) - set(ignore):
        df[col] -= df[col].min()
        df[col] /= df[col].max() - df[col].min() + offset
    return df


if __name__ == "__main__":
    np.random.seed(1)

    target = "saleprice"
    #junk = ["parcel"]

    df = {}
    X = {}
    Y = {}

    df["train"] = pd.read_csv("../data_sets/train.csv")
    df["validate"] = pd.read_csv("../data_sets/validate.csv")
    df["test"] = pd.read_csv("../data_sets/test.csv")

    '''
    for data in df:
        df[data] = df[data].drop(columns=junk)
        df[data] = df[data].astype(float)
    '''

    df["train"] = normalize(df["train"], ignore=[target])
    df["validate"] = normalize(df["validate"], df["train"], ignore=[target])
    df["test"] = normalize(df["test"], df["train"], ignore=[target])

    for data in df:
        X[data] = df[data].drop(columns=target)
        Y[data] = df[data][target]


    # Grid search.

    r2_best = 0
    lambda1best = None
    lambda2best = None
    W_final = None

    for lambda1 in [0.0001, 1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:

        model = Lasso(alpha=lambda1)
        model.fit(X["train"], Y["train"])
        r2 = model.score(X["validate"], Y["validate"])

        threshold = .01

        print("current: r^2={:0.4f}, lambda1={:0.4f}, weight_count={}"
              .format(r2, lambda1, len([w for w in model.coef_ if w > threshold])))

        colnames = df["train"].drop(columns=target).columns
        indices = [w > threshold for w in model.coef_]

        cols_above_thresh = list(colnames[indices])
        print(cols_above_thresh)

        '''
        if r2 > r2_best:
            r2_best = r2
            lambda1best = lambda1
            print("current best: r^2={:0.4f}, lambda1={:0.4f}".format(r2_best, lambda1best))
            #print("best coef: ", model.coef_)
        '''



