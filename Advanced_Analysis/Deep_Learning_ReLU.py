import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.models import Sequential
from sklearn import metrics
from tensorflow import set_random_seed


def normalize(df, on=None, ignore=(), offset=1e-5):
    on = on if on is not None else df
    for col in set(df.columns) - set(ignore):
        df[col] -= df[col].min()
        df[col] /= df[col].max() - df[col].min() + offset
        df[col] = 2 * df[col] - 1
    return df


def FFNN(X_train, Y_train, layer_widths, epochs):
    np.random.seed(1)
    set_random_seed(1)

    D = X_train.shape[1]

    model = Sequential()

    # Input layer.
    model.add(Dense(units=layer_widths[0], activation='tanh', input_shape=(D,)))

    # Hidden layers.
    for layer_width in layer_widths[1:]:
        model.add(Dense(units=layer_widths[1]))
        model.add(LeakyReLU(alpha=0.1))

    # Regression output layer.
    model.add(Dense(units=1))

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mse']
                  )

    model.fit(X_train, Y_train, epochs=epochs, batch_size=32)

    return model


if __name__ == "__main__":
    np.random.seed(1)

    df = {}
    X = {}
    Y = {}

    df["train"] = pd.read_csv("../data_sets/train.csv")
    df["validate"] = pd.read_csv("../data_sets/validate.csv")
    df["test"] = pd.read_csv("../data_sets/test.csv")

    target = "saleprice"

    df["train"] = normalize(df["train"], ignore=[target])
    df["validate"] = normalize(df["validate"], df["train"], ignore=[target])
    df["test"] = normalize(df["test"], df["train"], ignore=[target])

    for data in df:
        # remove largest
        df[data] = df[data].loc[df[data][target] < 3250000]

        #df[data] = df[data].drop(columns="mindisthosp/miles")

        X[data] = np.array(df[data].drop(columns=target))
        Y[data] = np.array(df[data][target])

        print(Y[data].max())

    # Grid search on kNN Regression hyperparameters.

    r2_best = 0
    model_best = None
    layer_widths_best = None

    for layer_widths in (
        # [400, 200, 50],
        # [100, 100, 50, 20],
        # [100, 200, 50, 20],
        # [200, 400, 100, 40],
        # [100, 20, 20, 20, 20],
        # [100, 50, 50, 20, 20],
        # [50, 100, 50, 20, 20],
        # [100, 100, 20, 20, 20],
        # [200, 100, 20, 20, 20],
        # [200, 400, 20, 20, 20],
        # [100, 200, 50, 50, 50],
        [100, 200, 20, 20, 20],
        [100, 400, 20, 20, 20],

    ):
        for epochs in [30, 40]:
            # Create the model.
            model = FFNN(X["train"], Y["train"], layer_widths, epochs=epochs)

            # Evaluate the model.
            Y_hat = model.predict(X["validate"])
            r2 = metrics.r2_score(y_true=Y["validate"], y_pred=Y_hat)

            print(f"R^2={r2}, layer_widths={layer_widths}")
            if r2 > r2_best:
                r2_best = r2
                layer_widths_best = layer_widths
                model_best = model

    print(f"Best model: R^2={r2_best}, layer_widths={layer_widths_best}, epochs={epochs}")
    print(model_best.summary())


    #Best model: R^2=0.8980244997617182, layer_widths=[100, 200, 20, 20, 20], epochs=40
    #Best model: test R^2=0.8937225695986469
    #Best model: R^2=0.9012138093923384, layer_widths=[100, 400, 20, 20, 20], epochs=30
    #Best model: test R^2=0.8678328955057804

    #Best model: R^2=0.8971884035349532, layer_widths=[100, 200, 20, 20, 20], epochs=40
    #Best model: test R^2=0.8874746424719424

    dpi = 300
    modelname = "FFNN"

    Y_hat = model_best.predict(X["test"])
    df_hat = df["test"].copy()
    df_hat[target] = Y["test"]
    df_hat["predicted"] = Y_hat
    df_hat["percent_residual"] = (df_hat["predicted"] - df_hat[target]) / df_hat[target]
    df_hat["neg_percent_residual"] = ((df_hat["predicted"] - df_hat[target]) / df_hat[target]).apply(lambda x: min(x, 0))
    df_hat["pos_percent_residual"] = ((df_hat["predicted"] - df_hat[target]) / df_hat[target]).apply(lambda x: max(x, 0))
    df_hat["abs_percent_residual"] = abs(df_hat["predicted"] - df_hat[target]) / df_hat[target]

    print(f"Best model: test R^2={metrics.r2_score(df_hat[target], df_hat['predicted'])}")

    plt.figure()
    plt.scatter(df_hat[target], df_hat["abs_percent_residual"], s=4)
    plt.xlim(0, 2e6)
    plt.ylim(0, 1)
    plt.title("FFNN Percent Absolute Error vs Actual Price (Capped at 100%)")
    plt.gca().set_xticklabels(['${:.2f}M'.format(x / 1e6) for x in plt.gca().get_xticks()])
    plt.xlabel("Actual Price")
    plt.gca().set_yticklabels(['{:.0f}%'.format(y * 100) for y in plt.gca().get_yticks()])
    plt.ylabel("Percent Absolute Error")
    plt.tight_layout()
    plt.savefig('figures/FFNN_Percent_Absolute_Error_vs_Sale_Price_restricted.png', dpi=dpi)



    df_hat["percent_residual"].to_csv(f"{modelname}_percent_residual.csv", index=False)
