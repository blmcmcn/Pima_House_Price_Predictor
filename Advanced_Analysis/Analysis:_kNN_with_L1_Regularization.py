import numpy as np
import pandas as pd
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


    # Hyperparameters.
    lambda1 = 1500
    k = 3
    norm = 1
    weights = "distance"
    threshold = .01

    # Perform L1 Regularization on Linear Regression to reduce data dimensionality.
    m_linear = Lasso(alpha=lambda1)
    m_linear.fit(X["train"], Y["train"])
    r2 = m_linear.score(X["validate"], Y["validate"])

    # Number of weights above threshold (including bias weight).
    count = len([w for w in m_linear.coef_ if w > threshold])

    print("linear regression: R^2={:0.4f}, lambda1={:0.4f}, weight_count={}"
          .format(r2, lambda1, len([w for w in m_linear.coef_ if w > threshold])))

    # Indices of columns that survive dimensionality reduction.
    indices = [i for i, w in enumerate(m_linear.coef_) if w > threshold]

    # Names of columns that survive dimensionality reduction.
    colnames = list(df["train"].drop(columns=target).columns[indices[1:]])
    print(colnames)

    # Reduced matrixes.
    for data in df:
        X[data + "_L1"] = X[data][:,indices]

        # Weighted metric.
        X[data + "_L1"][:,[10]] *= 3
        X[data + "_L1"][:,[4]] *= 3
        X[data + "_L1"][:,[12]] *= 3
        # R^2=0.8023933496394154


    # kNN Regression over reduced matrices.
    m_knn = KNeighborsRegressor(n_neighbors=k, p=norm, weights=weights)
    m_knn.fit(X["train_L1"], Y["train"])
    r2 = m_knn.score(X["validate_L1"], Y["validate"])
    print(f"kNN R^2={r2}, lambda1={lambda1}, count={count}, k={k}, norm={norm}, weights={weights}")





    # Analysis.
    '''
    errors = [(y["lat"], y["lon"], m_knn.predict(y.as_matrix(columns=colnames))) for y in Y["validate"]]
    '''
