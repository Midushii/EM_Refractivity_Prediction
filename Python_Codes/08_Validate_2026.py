"""
=========================================================
08_Validate_2026.py

Purpose:
Validate Random Forest predictions using
observed ERA5 refractivity for 2026.

Input:
    Predicted_2026.csv

Outputs:
    Validation_2026_Metrics.csv
    Observed_vs_Predicted_2026.png
    Time_Series_2026.png
    Residual_Plot_2026.png
    Prediction_Error_Histogram_2026.png
=========================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("="*60)
print("2026 EXTERNAL VALIDATION")
print("="*60)

# ---------------------------------------------------
# Read Prediction File
# ---------------------------------------------------

print("\nReading Predicted_2026.csv...")

df = pd.read_csv("Predicted_2026.csv")

print("Dataset Loaded")
print(df.shape)

observed = df["Radio_Refractivity"]
predicted = df["Predicted_Refractivity"]

# ---------------------------------------------------
# Metrics
# ---------------------------------------------------

mae = mean_absolute_error(observed, predicted)

rmse = np.sqrt(mean_squared_error(observed, predicted))

r2 = r2_score(observed, predicted)

bias = np.mean(predicted - observed)

print("\nValidation Metrics")
print("-"*40)

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")
print(f"Bias : {bias:.4f}")

# ---------------------------------------------------
# Error Statistics
# ---------------------------------------------------

error = predicted - observed

print("\nError Statistics")
print("-"*40)

print("Mean Error      :", round(error.mean(),3))
print("Std Error       :", round(error.std(),3))
print("Maximum Error   :", round(error.max(),3))
print("Minimum Error   :", round(error.min(),3))

# ---------------------------------------------------
# Save Metrics
# ---------------------------------------------------

metrics = pd.DataFrame({

    "Metric":[
        "MAE",
        "RMSE",
        "R2",
        "Bias",
        "Mean Error",
        "Std Error",
        "Maximum Error",
        "Minimum Error"
    ],

    "Value":[
        mae,
        rmse,
        r2,
        bias,
        error.mean(),
        error.std(),
        error.max(),
        error.min()
    ]

})

metrics.to_csv(
    "Validation_2026_Metrics.csv",
    index=False
)

# ---------------------------------------------------
# Scatter Plot
# ---------------------------------------------------

plt.figure(figsize=(7,7))

plt.scatter(
    observed,
    predicted,
    s=8
)

minimum=min(observed.min(),predicted.min())
maximum=max(observed.max(),predicted.max())

plt.plot(
    [minimum,maximum],
    [minimum,maximum]
)

plt.xlabel("Observed Refractivity (N)")
plt.ylabel("Predicted Refractivity (N)")
plt.title("Observed vs Predicted (2026)")

plt.tight_layout()

plt.savefig(
    "Observed_vs_Predicted_2026.png",
    dpi=300
)

plt.close()

# ---------------------------------------------------
# Time Series
# ---------------------------------------------------

plt.figure(figsize=(12,5))

plt.plot(
    observed.values,
    label="Observed"
)

plt.plot(
    predicted.values,
    label="Predicted"
)

plt.legend()

plt.xlabel("Sample")

plt.ylabel("Radio Refractivity")

plt.title("Observed vs Predicted Time Series (2026)")

plt.tight_layout()

plt.savefig(
    "Time_Series_2026.png",
    dpi=300
)

plt.close()

# ---------------------------------------------------
# Histogram
# ---------------------------------------------------

plt.figure(figsize=(7,5))

plt.hist(
    error,
    bins=40
)

plt.xlabel("Prediction Error")

plt.ylabel("Frequency")

plt.title("Prediction Error Distribution")

plt.tight_layout()

plt.savefig(
    "Prediction_Error_Histogram_2026.png",
    dpi=300
)

plt.close()

# ---------------------------------------------------
# Residual Plot
# ---------------------------------------------------

plt.figure(figsize=(7,5))

plt.scatter(
    predicted,
    error,
    s=8
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Predicted Refractivity")

plt.ylabel("Residual")

plt.title("Residual Plot")

plt.tight_layout()

plt.savefig(
    "Residual_Plot_2026.png",
    dpi=300
)

plt.close()

print("\nValidation Complete")

print("-"*40)

print("Files Generated")

print("✓ Validation_2026_Metrics.csv")
print("✓ Observed_vs_Predicted_2026.png")
print("✓ Time_Series_2026.png")
print("✓ Prediction_Error_Histogram_2026.png")
print("✓ Residual_Plot_2026.png")

print("="*60)