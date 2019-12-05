import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dpi = 300

df = pd.read_csv("importances.csv")
df = df.transpose()
df.columns = df.loc["feature"]
df = df.loc["importance"]
print(df)

df_plot = pd.Series()


# print all features with lat or lon
latlons = []
for feature in df.index:
    if "lat" in feature or "lon" in feature:
        latlons.append(feature)
print(latlons)

# get their sum

#df_plot["Tax\nAssessment"] = df["actual"]
df_plot["House SqFt"] = df["sqft"]
df_plot["Land SqFt"] = df["landsqft"]
df_plot["Year"] = df["recordingdate"] + df["year"] + df["saledate"]
df_plot["Openness"] = df["openness"]
df_plot["Latitude/\nLongitude"] = df[latlons].sum()
df_plot["Altitude"] = df["z"]
df_plot["Rooms"] = df["rooms"]
df_plot["Tax\nAdjustment"] = df["adjustment"]

df_plot = df_plot.sort_values()[::-1]

print(df_plot)

plt.figure()
plt.bar(range(df_plot.shape[0]), df_plot)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.xticks(range(df_plot.shape[0]), df_plot.index, rotation=90)
plt.title("RF Feature Importance")
plt.tight_layout()
plt.savefig(f"figures/RF_Feature_Importance.png", dpi=dpi)
