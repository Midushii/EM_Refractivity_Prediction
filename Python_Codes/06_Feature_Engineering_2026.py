import pandas as pd
import numpy as np

print("="*60)
print("FEATURE ENGINEERING FOR 2026")
print("="*60)

print("\nReading ML_Dataset_2026.csv...")

df = pd.read_csv("ML_Dataset_2026.csv")

print("Dataset Loaded")

print(df.shape)

date = pd.to_datetime(
    dict(
        year=df.Year,
        month=df.Month,
        day=df.Day
    )
)

df["Day_of_Year"] = date.dt.dayofyear

df["Month_Sin"] = np.sin(
    2*np.pi*df["Month"]/12
)

df["Month_Cos"] = np.cos(
    2*np.pi*df["Month"]/12
)

df["Hour_Sin"] = np.sin(
    2*np.pi*df["Hour"]/24
)

df["Hour_Cos"] = np.cos(
    2*np.pi*df["Hour"]/24
)

def season(month):

    if month in [12,1,2]:
        return 1

    elif month in [3,4,5]:
        return 2

    elif month in [6,7,8]:
        return 3

    else:
        return 4

df["Season"] = df["Month"].apply(season)

df.to_csv(
    "ML_Features_2026.csv",
    index=False
)

print("\nFeatures Created Successfully")

print(df.head())

print("\nTotal Samples :",len(df))

print("\nSaved as ML_Features_2026.csv")

print("="*60)
