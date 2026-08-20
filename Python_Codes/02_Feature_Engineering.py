import pandas as pd
import numpy as np

print("=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

print("\nReading ML_Dataset.csv ...")

df = pd.read_csv("ML_Dataset.csv")

print("Dataset Loaded Successfully")
print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")


print("\nCreating Date Information...")

df["Date"] = pd.to_datetime(
    dict(
        year=df["Year"],
        month=df["Month"],
        day=df["Day"]
    )
)

df["Day_of_Year"] = df["Date"].dt.dayofyear

df["Month_Sin"] = np.sin(
    2 * np.pi * df["Month"] / 12
)

df["Month_Cos"] = np.cos(
    2 * np.pi * df["Month"] / 12
)


df["Hour_Sin"] = np.sin(
    2 * np.pi * df["Hour"] / 24
)

df["Hour_Cos"] = np.cos(
    2 * np.pi * df["Hour"] / 24
)

def season(month):

    if month in [12,1,2]:
        return "Winter"

    elif month in [3,4,5]:
        return "Summer"

    elif month in [6,7,8,9]:
        return "Monsoon"

    else:
        return "Post-Monsoon"

df["Season"] = df["Month"].apply(season)

print("\nComputing Feature Correlation...")

numeric_df = df.select_dtypes(include=np.number)

corr = numeric_df.corr()

corr.to_csv("Feature_Correlation.csv")


df.drop(columns=["Date"], inplace=True)


df.to_csv("ML_Features.csv", index=False)
print("\n")
print("=" * 60)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 60)

print("\nFinal Dataset Shape")

print(df.shape)

print("\nFeature List\n")

for col in df.columns:
    print("-", col)

print("\nFirst Five Rows\n")

print(df.head())

print("\nFiles Created")

print("✓ ML_Features.csv")
print("✓ Feature_Correlation.csv")

print("\nDone.")
