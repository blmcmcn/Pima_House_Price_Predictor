import pandas as pd

df = pd.read_csv("y_hat.csv")
df["residual"] = df["predicted"] - df["saleprice"]
df["percent_residual"] = (df["predicted"] - df["saleprice"]) / df["saleprice"]

print("residual std: ", df["residual"].std())
print("percent_residual std: ", df["percent_residual"].std())

# residual std:  44510.605796389806
# percent_residual std:  0.1660785986334228
