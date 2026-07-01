"""
=========================================================
02_Feature_Engineering.py

Purpose:
Prepare the Machine Learning dataset by creating
time-based cyclical features for predicting
radio refractivity.

Input:
    ML_Dataset.csv

Output:
    ML_Features.csv
    Feature_Correlation.csv
=========================================================
"""

import pandas as pd
import numpy as np

print("=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# -------------------------------------------------------
# Read Dataset
# -------------------------------------------------------

print("\nReading ML_Dataset.csv ...")

df = pd.read_csv("ML_Dataset.csv")

print("Dataset Loaded Successfully")
print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")

# -------------------------------------------------------
# Create Date Column
# -------------------------------------------------------

print("\nCreating Date Information...")

df["Date"] = pd.to_datetime(
    dict(
        year=df["Year"],
        month=df["Month"],
        day=df["Day"]
    )
)

# -------------------------------------------------------
# Day of Year
# -------------------------------------------------------

df["Day_of_Year"] = df["Date"].dt.dayofyear

# -------------------------------------------------------
# Month Cyclic Encoding
# -------------------------------------------------------

df["Month_Sin"] = np.sin(
    2 * np.pi * df["Month"] / 12
)

df["Month_Cos"] = np.cos(
    2 * np.pi * df["Month"] / 12
)

# -------------------------------------------------------
# Hour Cyclic Encoding
# -------------------------------------------------------

df["Hour_Sin"] = np.sin(
    2 * np.pi * df["Hour"] / 24
)

df["Hour_Cos"] = np.cos(
    2 * np.pi * df["Hour"] / 24
)

# -------------------------------------------------------
# Season
# -------------------------------------------------------

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

# -------------------------------------------------------
# Feature Correlation
# -------------------------------------------------------

print("\nComputing Feature Correlation...")

numeric_df = df.select_dtypes(include=np.number)

corr = numeric_df.corr()

corr.to_csv("Feature_Correlation.csv")

# -------------------------------------------------------
# Remove Temporary Date Column
# -------------------------------------------------------

df.drop(columns=["Date"], inplace=True)

# -------------------------------------------------------
# Save Feature Dataset
# -------------------------------------------------------

df.to_csv("ML_Features.csv", index=False)

# -------------------------------------------------------
# Display Information
# -------------------------------------------------------

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