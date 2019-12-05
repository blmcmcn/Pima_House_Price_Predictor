import pandas as pd

def check_categories(filename):

    print("\nchecking for categories...")

    # load data
    df = pd.read_csv(filename)

    # Number of observations.
    N,D = df.shape

    fractions = []

    for col in df.columns:
        # Get categorical columns.
        if set(df[col]) <= {0, 1}:
            # (column name, amount true)
            S = df[col].sum()
            fractions.append( (col, S/N, S) )

    fractions.sort(key=lambda x: x[1])

    print("\nCategorical Data\nPercentage : N_observations : Category")
    for i in fractions:
        print("{:0.4f} : {} : {}".format(i[1]*100,i[2],i[0]))
