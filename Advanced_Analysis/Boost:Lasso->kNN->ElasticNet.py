import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score


# Data preparation.

def normalize(df, on=None, ignore=(), offset=1e-5):
    on = on if on is not None else df
    for col in set(df.columns) - set(ignore):
        df[col] -= df[col].min()
        df[col] /= df[col].max() - df[col].min() + offset
    return df


class Linear_Boosted_kNN(object):
    def __init__(self, lambda1, lambda2):
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # Optimal L1-Regularized kNN model.
        L1kNN_lambda1 = 600
        L1kNN_n = 5
        L1kNN_metric = 2
        L1kNN_weights = 'distance'

        L1Regression = Lasso(alpha=L1kNN_lambda1)
        L1Regression.fit(X["train"], Y["train"])

        threshold = .01

        count = len([w for w in L1Regression.coef_ if w > threshold])
        self.indices = [i for i, w in enumerate(L1Regression.coef_) if w > threshold]

        L1kNN = KNeighborsRegressor(n_neighbors=L1kNN_n, p=L1kNN_metric, weights=L1kNN_weights)
        L1kNN.fit(X["train"][:, self.indices], Y["train"])

        self.L1kNN = L1kNN

    def residuals(self, X, Y, epsilon=1e-10):
        residuals = self.L1kNN.predict(X[:, self.indices] + epsilon) - Y
        #print(sorted(list(residuals)))
        return residuals

    def fit(self, X, Y):
        alpha = self.lambda1 + self.lambda2
        l1_ratio = self.lambda1 / (self.lambda1 + self.lambda2)

        self.ElasticNet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

        # Train on the residuals left by the L1-Regularized kNN model.
        #print(self.residuals(X, Y))
        self.ElasticNet.fit(X, 10000*self.residuals(X, Y))

    def predict(self, X):
        y_hat_L1kNN = self.L1kNN.predict(X[:, self.indices])
        y_hat_L1kNN = y_hat_L1kNN.reshape((y_hat_L1kNN.shape[0],))
        y_hat_ElasticNet = self.ElasticNet.predict(X)
        return y_hat_L1kNN + y_hat_ElasticNet

    def score(self, X, Y):
        return r2_score(Y, self.predict(X))


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
    lambda2_best = None

    for lambda1 in [.1, .5, 1, 5, 10, 50, 100]:
        for lambda2 in [.1, .5, 10000, 5, 10, 50, 100]:

            model = Linear_Boosted_kNN(lambda1, lambda2)

            model.fit(X["train"], Y["train"])

            r2 = model.score(X["validate"], Y["validate"])

            print(f"boosted R^2={r2}, lambda1={lambda1}, lambda2={lambda2}")
            if r2 > r2_best:
                r2_best = r2
                lambda1_best = lambda1
                lambda2_best = lambda2


    print(f"best boosted validation R^2={r2_best}, lambda1={lambda1_best}, lambda2={lambda2_best}")

    # test_r2 = model_best.score(X["test"], Y["test"])
    # print(f"boosted test R^2={r2_best}, lambda1={lambda1_best}, lambda2={lambda2_best}")

