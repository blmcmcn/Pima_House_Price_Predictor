import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor


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

    df = {}
    X = {}
    Y = {}
    X_pca = {}

    df["train"] = pd.read_csv("../data_sets/train.csv")
    df["validate"] = pd.read_csv("../data_sets/validate.csv")
    df["test"] = pd.read_csv("../data_sets/test.csv")

    df["train"] = normalize(df["train"], ignore=[target])
    df["validate"] = normalize(df["validate"], df["train"], ignore=[target])
    df["test"] = normalize(df["test"], df["train"], ignore=[target])

    for data in df:
        X[data] = np.array(df[data].drop(columns=target))
        Y[data] = df[data].as_matrix(columns=[target])


    # Grid search.

    r2_best = 0
    var_best = None
    count_best = None
    n_best = None
    norm_best = None
    weights_best = None

    for var in [.95, .98, .99, .995, .999]:

        pca = PCA(n_components=var)
        pca.fit(X["train"])

        X_pca = {}
        for data in df:
            X_pca[data] = pca.transform(X[data])

        count = pca.n_components_

        for n in range(3, 15):
            for norm in [1, 2]:
                for weights in ['uniform', 'distance']:

                    model = KNeighborsRegressor(n_neighbors=n, p=norm, weights=weights)
                    model.fit(X_pca["train"], Y["train"])
                    r2 = model.score(X_pca["validate"], Y["validate"])
                    print(f"kNN R^2={r2}, var={var}, count={count}, n={n}, norm={norm}, weights={weights}")
                    if r2 > r2_best:
                        r2_best = r2
                        var_best = var
                        count_best = count
                        n_best = n
                        norm_best = norm
                        weights_best = weights

    print(f"Best model: R^2={r2_best}, var={var_best}, count={count_best}, n={n_best}, norm={norm_best}, weights={weights_best}")

    '''
    # Test set R^2.
    test_r2 = model.score(X_pca["test"], Y["test"])
    print(f"Best model: validation R^2={r2_best}, test R^2={test_r2}, lambda1={lambda1_best}, n={n_best}, norm={norm_best}, weights={weights_best}")
    '''

    # best
    #Best model: R^2=0.748043370586482, var=0.999, count=53, n=5, norm=1, weights=distance
