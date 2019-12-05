import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor



def normalize(df, on=None, ignore=(), offset=1e-5):
    on = on if on is not None else df
    for col in set(df.columns) - set(ignore):
        df[col] -= df[col].min()
        df[col] /= df[col].max() - df[col].min() + offset
    return df


if __name__ == "__main__":
    target = "saleprice"

    df = {}
    X = {}
    Y = {}

    df["train"] = pd.read_csv("../data_sets/train.csv")
    df["validate"] = pd.read_csv("../data_sets/validate.csv")
    df["test"] = pd.read_csv("../data_sets/test.csv")

    df["train"] = normalize(df["train"], ignore=target)
    df["validate"] = normalize(df["validate"], df["train"], ignore=target)
    df["test"] = normalize(df["test"], df["train"], ignore=target)

    for data in df:
        X[data] = np.array(df[data].drop(columns=target))
        Y[data] = np.array(df[data][target])


    # Grid search on kNN Regression hyperparameters.

    r2_best = 0
    n_best = None
    norm_best = None
    weights_best = None

    for n in range(2, 20):
        for norm in [1, 2]:
            for weights in ['uniform', 'distance']:
                model = KNeighborsRegressor(n_neighbors=n, p=norm, weights=weights)
                model.fit(X["train"], Y["train"])
                r2 = model.score(X["validate"], Y["validate"])
                print(f"R^2={r2}, n={n}, norm={norm}, weights={weights}")
                if r2 > r2_best:
                    r2_best = r2
                    n_best = n
                    norm_best = norm
                    weights_best = weights

    print(f"Best model: R^2={r2_best}, n={n_best}, norm={norm_best}, weights={weights_best}")
