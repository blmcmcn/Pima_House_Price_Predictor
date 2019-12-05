import numpy as np
import pandas as pd
from pyswarms.single.global_best import GlobalBestPSO
from sklearn.linear_model import Lasso
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
    model_best = None
    lambda1_best = None
    n_best = None
    norm_best = None
    indices_best = None
    weights_best = None

    for lambda1 in [600]:
    #for lambda1 in [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]:

        model = Lasso(alpha=lambda1)
        model.fit(X["train"], Y["train"])
        r2 = model.score(X["validate"], Y["validate"])

        threshold = .01

        count = len([w for w in model.coef_ if w > threshold])


        print("linear regression: R^2={:0.4f}, lambda1={:0.4f}, weight_count={}"
              .format(r2, lambda1, len([w for w in model.coef_ if w > threshold])))

        '''
        colnames = df["train"].drop(columns=target).columns
        indices = [0] + [w > threshold for w in model.coef_]

        cols_above_thresh = list(colnames[indices])
        '''

        indices = [i for i, w in enumerate(model.coef_) if w > threshold]
        print('indices: ', indices)

        for n in [5]:
        #for n in range(3, 15):
            for norm in [2]:
            #for norm in [1, 2]:
                for weights in ['distance']:
                #for weights in ['uniform', 'distance']:
                    model = KNeighborsRegressor(n_neighbors=n, p=norm, weights=weights)
                    model.fit(X["train"][:,indices], Y["train"])
                    r2 = model.score(X["validate"][:,indices], Y["validate"])
                    print(f"kNN R^2={r2}, lambda1={lambda1}, count={count}, n={n}, norm={norm}, weights={weights}")
                    if r2 > r2_best:
                        r2_best = r2
                        lambda1_best = lambda1
                        n_best = n
                        norm_best = norm
                        weights_best = weights
                        indices_best = indices
                        model_best = model

    def swarm_objective():
        r2 = model.score(X["validate"][:, indices], Y["validate"])
        return r2

    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    optimizer = GlobalBestPSO(n_particles=10, dimensions=2, options=options)
    cost, pos = optimizer.optimize(swarm_objective, 1000, print_step=100, verbose=3, a=1, b=100, c=0)

    test_r2 = model.score(X["validate"][:,indices], Y["validate"])
    print(f"Best model: validation R^2={r2_best}, test R^2={test_r2}, lambda1={lambda1_best}, n={n_best}, norm={norm_best}, weights={weights_best}")


    # best
    #Best model: validation R^2=0.7936490391438129, test R^2=0.7669037305153521, lambda1=600, n=5, norm=2, weights=distance
