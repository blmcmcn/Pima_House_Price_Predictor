import matplotlib.pyplot as plt
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

    for data in df:
        # drop largest
        df[data] = df[data].loc[df[data][target] < 3250000]

        # drop geographic
        df[data] = df[data].drop(columns="mindisthosp/miles")

        # add openness feature
        df[data]["openness"] = df[data]["sqft"] / (df[data]["rooms"] + 1e-8)

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

    for lambda1 in [900]:
    #for lambda1 in [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]:
        np.random.seed(1)

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

        for n in [9]:
        #for n in range(3, 15):
            for norm in [1]:
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

    test_r2 = model.score(X["test"][:,indices], Y["test"])
    print(f"Best model: validation R^2={r2_best}, test R^2={test_r2}, lambda1={lambda1_best}, n={n_best}, norm={norm_best}, weights={weights_best}")


    #Best model: validation R^2=0.8865443206833918, test R^2=0.8599163595442789, lambda1=900, n=9, norm=1, weights=distance

    dpi = 300
    modelname = "kNN w/ L1"
    model_name = "kNN_L1"

    Y_hat = model_best.predict(X["test"][:,indices])

    df_hat = df["test"].copy()
    df_hat[target] = Y["test"]
    df_hat["predicted"] = Y_hat
    df_hat["percent_residual"] = (df_hat["predicted"] - df_hat[target]) / df_hat[target]
    df_hat["neg_percent_residual"] = ((df_hat["predicted"] - df_hat[target]) / df_hat[target]).apply(lambda x: min(x, 0))
    df_hat["pos_percent_residual"] = ((df_hat["predicted"] - df_hat[target]) / df_hat[target]).apply(lambda x: max(x, 0))
    df_hat["abs_percent_residual"] = abs(df_hat["predicted"] - df_hat[target]) / df_hat[target]

    df_hat["percent_residual"].to_csv(f"{model_name}_percent_residual.csv", index=False)

    plt.figure()
    plt.scatter(Y["test"], Y_hat, s=4)
    plt.plot(range(0, Y["test"].max()), range(0, Y["test"].max()), 'k--')
    plt.gca().set_xticklabels(['${:.2f}M'.format(x / 1e6) for x in plt.gca().get_xticks()])
    plt.xlabel("Predicted Price")
    plt.gca().set_yticklabels(['${:.2f}M'.format(y / 1e6) for y in plt.gca().get_yticks()])
    plt.ylabel("Actual Price")
    plt.title(f"{modelname} Predicted Price vs Actual Price")
    plt.tight_layout()
    plt.savefig(f"figures/{model_name}_Predicted_vs_Sale_Price.png", dpi=dpi)

    plt.figure()
    plt.scatter(df_hat[target], df_hat["abs_percent_residual"], s=4)
    plt.xlim(0, 2e6)
    plt.ylim(0, 1)
    plt.gca().set_xticklabels(['${:.2f}M'.format(x / 1e6) for x in plt.gca().get_xticks()])
    plt.xlabel("Actual Price")
    plt.gca().set_yticklabels(['{:.0f}%'.format(y * 100) for y in plt.gca().get_yticks()])
    plt.ylabel("Absolute Percent Error")
    plt.title(f"{modelname} Absolute Percent Error vs Actual Price (Capped at 100%)")
    plt.tight_layout()
    plt.savefig(f"figures/{model_name}_Absolute_Percent_Error_vs_Sale_Price_restricted.png", dpi=dpi)
